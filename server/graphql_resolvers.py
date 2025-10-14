"""
GraphQL Resolvers for Redactify
Implements the business logic for GraphQL operations
"""

import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from graphql_schema import (
    AnonymizationResult,
    BatchAnonymizationResult,
    DetectionResult,
    HealthStatus,
    MCPServersHealth,
    SystemConfig,
    Entity,
    Metadata,
    ComponentHealth,
    MCPServerStatus,
    PIIOptionsInput,
    HealthStatusEnum,
    AnonymizationStrategy,
)

# Import existing engines
from detection_engine import get_detection_engine
from anonymization_engine import get_anonymization_engine
from model_manager import get_model_manager
from auto_mcp_manager import get_auto_mcp_manager

logger = logging.getLogger("GraphQLResolvers")

# Get singleton instances
detection_engine = get_detection_engine()
anonymization_engine = get_anonymization_engine()
model_manager = get_model_manager()
auto_mcp_manager = get_auto_mcp_manager()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_entity_to_graphql(entity: Dict[str, Any]) -> Entity:
    """Convert backend entity format to GraphQL Entity type"""
    return Entity(
        entity_group=entity.get('entity_group', 'UNKNOWN'),
        word=entity.get('word', ''),
        start=entity.get('start', 0),
        end=entity.get('end', 0),
        score=float(entity.get('score', 0.0)),
        detector=entity.get('detector', 'unknown')
    )


def convert_pii_options(options: Optional[PIIOptionsInput]) -> Optional[Dict[str, bool]]:
    """Convert GraphQL PIIOptionsInput to backend format"""
    if options is None:
        return None
    return options.to_dict()


def validate_text_input(text: str) -> None:
    """Validate text input"""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) > 1_000_000:  # 1MB limit
        raise ValueError("Text exceeds maximum length of 1,000,000 characters")


def validate_batch_input(texts: List[str]) -> None:
    """Validate batch input"""
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    if len(texts) > 100:  # Batch size limit
        raise ValueError("Batch size exceeds maximum of 100 texts")
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Text at index {i} is empty")


# ============================================================================
# QUERY RESOLVERS
# ============================================================================

async def resolve_health() -> HealthStatus:
    """Resolve health check query"""
    try:
        logger.info("GraphQL: Resolving health query")
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "detection_engine": "ok",
                "anonymization_engine": "ok",
                "model_manager": "ok",
                "mcp_servers": "checking..."
            }
        }
        
        # Check model manager
        try:
            stats = model_manager.get_model_stats()
            health_status["components"]["model_manager"] = "ok"
            health_status["loaded_models"] = stats.get("loaded_models", 0)
        except Exception as e:
            health_status["components"]["model_manager"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check MCP servers
        try:
            mcp_health = await auto_mcp_manager.check_all_health()
            healthy_servers = sum(1 for is_healthy in mcp_health.values() if is_healthy)
            total_servers = len(mcp_health)
            
            server_status = auto_mcp_manager.get_server_status()
            mcp_servers_list = []
            
            for name, status in server_status.items():
                mcp_servers_list.append(MCPServerStatus(
                    name=name,
                    running=status.get("running", False),
                    healthy=mcp_health.get(name, False),
                    port=status.get("port", 0),
                    url=f"http://localhost:{status.get('port', 0)}",
                    pid=status.get("pid"),
                    uptime=status.get("uptime")
                ))
            
            health_status["mcp_servers"] = MCPServersHealth(
                healthy=healthy_servers,
                total=total_servers,
                servers=mcp_servers_list
            )
            
            health_status["components"]["mcp_servers"] = f"{healthy_servers}/{total_servers} healthy"
            
            if healthy_servers == 0:
                health_status["status"] = "degraded"
            elif healthy_servers < total_servers:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["mcp_servers"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Convert to GraphQL types
        return HealthStatus(
            status=HealthStatusEnum[health_status["status"].upper()],
            timestamp=health_status["timestamp"],
            components=ComponentHealth(
                detection_engine=health_status["components"]["detection_engine"],
                anonymization_engine=health_status["components"]["anonymization_engine"],
                model_manager=health_status["components"]["model_manager"],
                mcp_servers=health_status["components"]["mcp_servers"]
            ),
            loaded_models=health_status.get("loaded_models"),
            mcp_servers=health_status.get("mcp_servers")
        )
        
    except Exception as e:
        logger.error(f"Error in health resolver: {e}", exc_info=True)
        return HealthStatus(
            status=HealthStatusEnum.UNHEALTHY,
            timestamp=datetime.utcnow().isoformat(),
            components=ComponentHealth(
                detection_engine="error",
                anonymization_engine="error",
                model_manager="error",
                mcp_servers="error"
            )
        )


async def resolve_mcp_status() -> MCPServersHealth:
    """Resolve MCP status query"""
    try:
        logger.info("GraphQL: Resolving MCP status query")
        
        server_status = auto_mcp_manager.get_server_status()
        health_results = await auto_mcp_manager.check_all_health()
        
        servers = []
        for name, status in server_status.items():
            servers.append(MCPServerStatus(
                name=name,
                running=status.get("running", False),
                healthy=health_results.get(name, False),
                port=status.get("port", 0),
                url=f"http://localhost:{status.get('port', 0)}",
                pid=status.get("pid"),
                uptime=status.get("uptime")
            ))
        
        healthy_count = sum(1 for s in servers if s.healthy)
        
        return MCPServersHealth(
            healthy=healthy_count,
            total=len(servers),
            servers=servers
        )
        
    except Exception as e:
        logger.error(f"Error in MCP status resolver: {e}", exc_info=True)
        return MCPServersHealth(healthy=0, total=0, servers=[])


async def resolve_config() -> SystemConfig:
    """Resolve config query"""
    try:
        logger.info("GraphQL: Resolving config query")
        
        return SystemConfig(
            version="2.0",
            architecture="MCP",
            available_models=model_manager.get_available_models(),
            supported_entities=[
                "PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS",
                "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS",
                "URL", "DATE_TIME", "PASSWORD", "API_KEY", "ROLL_NUMBER",
                "DEPLOY_TOKEN", "AUTHENTICATION", "FINANCIAL", "CREDENTIAL",
                "DEVICE", "MEDICAL", "ID_NUMBER"
            ],
            anonymization_strategies=["pseudonymize", "mask", "redact", "custom"],
            model_stats=model_manager.get_model_stats()
        )
        
    except Exception as e:
        logger.error(f"Error in config resolver: {e}", exc_info=True)
        raise


async def resolve_detect_entities(
    text: str,
    options: Optional[PIIOptionsInput] = None
) -> DetectionResult:
    """Resolve entity detection query"""
    try:
        logger.info(f"GraphQL: Resolving detect_entities query (text length: {len(text)})")
        
        # Validate input
        validate_text_input(text)
        
        start_time = time.time()
        
        # Detect entities
        entities, domains = await detection_engine.detect_entities(text)
        
        # Normalize entity types
        normalized_entities = []
        for entity in entities:
            entity_type = entity.get('entity_group', '').upper()
            
            if entity_type in ['PER', 'PERSON']:
                entity['entity_group'] = 'PERSON'
            elif entity_type in ['ORG', 'ORGANIZATION']:
                entity['entity_group'] = 'ORGANIZATION'
            elif entity_type in ['LOC', 'LOCATION']:
                entity['entity_group'] = 'LOCATION'
            elif entity_type == 'MISC':
                continue
            
            normalized_entities.append(entity)
        
        # Filter by options if provided
        if options:
            options_dict = convert_pii_options(options)
            if options_dict:
                filtered_entities = []
                for entity in normalized_entities:
                    entity_type = entity.get('entity_group', '').upper()
                    if options_dict.get(entity_type, True):
                        filtered_entities.append(entity)
                normalized_entities = filtered_entities
        
        processing_time = time.time() - start_time
        
        # Convert to GraphQL types
        graphql_entities = [convert_entity_to_graphql(e) for e in normalized_entities]
        entity_types = list(set(e.entity_group for e in graphql_entities))
        
        return DetectionResult(
            entities=graphql_entities,
            domains_detected=domains,
            total_entities=len(graphql_entities),
            entity_types=entity_types,
            processing_time=processing_time
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in detect_entities: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in detect_entities resolver: {e}", exc_info=True)
        raise


# ============================================================================
# MUTATION RESOLVERS
# ============================================================================

async def resolve_anonymize(
    text: str,
    options: Optional[PIIOptionsInput] = None,
    full_redaction: bool = True
) -> AnonymizationResult:
    """Resolve anonymize mutation"""
    try:
        logger.info(f"GraphQL: Resolving anonymize mutation (text length: {len(text)})")
        
        # Validate input
        validate_text_input(text)
        
        start_time = time.time()
        
        # Step 1: Detect entities
        entities, domains = await detection_engine.detect_entities(text)
        
        # Step 2: Normalize entity types
        normalized_entities = []
        for entity in entities:
            entity_type = entity.get('entity_group', '').upper()
            
            if entity_type in ['PER', 'PERSON']:
                entity['entity_group'] = 'PERSON'
            elif entity_type in ['ORG', 'ORGANIZATION']:
                entity['entity_group'] = 'ORGANIZATION'
            elif entity_type in ['LOC', 'LOCATION']:
                entity['entity_group'] = 'LOCATION'
            elif entity_type == 'MISC':
                continue
            
            normalized_entities.append(entity)
        
        # Step 3: Filter by options if provided
        if options:
            options_dict = convert_pii_options(options)
            if options_dict:
                filtered_entities = []
                for entity in normalized_entities:
                    entity_type = entity.get('entity_group', '').upper()
                    if options_dict.get(entity_type, True):
                        filtered_entities.append(entity)
                normalized_entities = filtered_entities
        
        # Step 4: Anonymize text
        strategy = "pseudonymize" if full_redaction else "mask"
        anonymization_result = anonymization_engine.anonymize_text(
            text=text,
            entities=normalized_entities,
            strategy=strategy,
            preserve_format=not full_redaction
        )
        
        processing_time = time.time() - start_time
        
        # Convert to GraphQL types
        graphql_entities = [convert_entity_to_graphql(e) for e in normalized_entities]
        detectors_used = list(set(e.detector for e in graphql_entities))
        
        return AnonymizationResult(
            anonymized_text=anonymization_result["anonymized_text"],
            entities=graphql_entities,
            processing_time=processing_time,
            domains_detected=domains,
            entities_processed=len(anonymization_result["entities_processed"]),
            strategy_used=AnonymizationStrategy[strategy.upper()],
            metadata=Metadata(
                total_entities=len(graphql_entities),
                domains_used=domains,
                detectors_used=detectors_used
            )
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in anonymize: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in anonymize resolver: {e}", exc_info=True)
        raise


async def resolve_anonymize_batch(
    texts: List[str],
    options: Optional[PIIOptionsInput] = None,
    full_redaction: bool = True
) -> BatchAnonymizationResult:
    """Resolve batch anonymize mutation"""
    try:
        logger.info(f"GraphQL: Resolving anonymize_batch mutation (batch size: {len(texts)})")
        
        # Validate input
        validate_batch_input(texts)
        
        start_time = time.time()
        
        # Batch detect entities
        batch_results = await detection_engine.detect_entities_batch(texts)
        
        # Process each result
        strategy = "pseudonymize" if full_redaction else "mask"
        results = []
        
        for i, (entities, domains) in enumerate(batch_results):
            text = texts[i]
            
            # Normalize entity types
            normalized_entities = []
            for entity in entities:
                entity_type = entity.get('entity_group', '').upper()
                
                if entity_type in ['PER', 'PERSON']:
                    entity['entity_group'] = 'PERSON'
                elif entity_type in ['ORG', 'ORGANIZATION']:
                    entity['entity_group'] = 'ORGANIZATION'
                elif entity_type in ['LOC', 'LOCATION']:
                    entity['entity_group'] = 'LOCATION'
                elif entity_type == 'MISC':
                    continue
                
                normalized_entities.append(entity)
            
            # Filter by options if provided
            if options:
                options_dict = convert_pii_options(options)
                if options_dict:
                    filtered_entities = []
                    for entity in normalized_entities:
                        entity_type = entity.get('entity_group', '').upper()
                        if options_dict.get(entity_type, True):
                            filtered_entities.append(entity)
                    normalized_entities = filtered_entities
            
            # Anonymize text
            anonymization_result = anonymization_engine.anonymize_text(
                text=text,
                entities=normalized_entities,
                strategy=strategy,
                preserve_format=not full_redaction
            )
            
            # Convert to GraphQL types
            graphql_entities = [convert_entity_to_graphql(e) for e in normalized_entities]
            detectors_used = list(set(e.detector for e in graphql_entities))
            
            results.append(AnonymizationResult(
                anonymized_text=anonymization_result["anonymized_text"],
                entities=graphql_entities,
                processing_time=0.0,  # Individual timing not tracked in batch
                domains_detected=domains,
                entities_processed=len(anonymization_result["entities_processed"]),
                strategy_used=AnonymizationStrategy[strategy.upper()],
                metadata=Metadata(
                    total_entities=len(graphql_entities),
                    domains_used=domains,
                    detectors_used=detectors_used
                )
            ))
        
        total_processing_time = time.time() - start_time
        
        return BatchAnonymizationResult(
            results=results,
            batch_size=len(texts),
            total_processing_time=total_processing_time,
            average_time_per_text=total_processing_time / len(texts) if texts else 0,
            strategy_used=AnonymizationStrategy[strategy.upper()]
        )
        
    except ValueError as e:
        logger.warning(f"Validation error in anonymize_batch: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in anonymize_batch resolver: {e}", exc_info=True)
        raise


async def resolve_anonymize_with_preview(
    text: str,
    options: Optional[PIIOptionsInput] = None
) -> DetectionResult:
    """Resolve anonymize with preview mutation (detection only)"""
    # This is essentially the same as detect_entities
    return await resolve_detect_entities(text, options)
