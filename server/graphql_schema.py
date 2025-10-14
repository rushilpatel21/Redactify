"""
GraphQL Schema for Redactify
Using Strawberry GraphQL for type-safe, modern GraphQL implementation
"""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS - For type safety and validation
# ============================================================================

@strawberry.enum
class AnonymizationStrategy(Enum):
    """Anonymization strategy options"""
    PSEUDONYMIZE = "pseudonymize"
    MASK = "mask"
    REDACT = "redact"
    CUSTOM = "custom"


@strawberry.enum
class HealthStatusEnum(Enum):
    """Health status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@strawberry.enum
class PIITypeEnum(Enum):
    """All supported PII types - ensures client can't request invalid types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    DATE_TIME = "DATE_TIME"
    PASSWORD = "PASSWORD"
    API_KEY = "API_KEY"
    ROLL_NUMBER = "ROLL_NUMBER"
    DEPLOY_TOKEN = "DEPLOY_TOKEN"
    AUTHENTICATION = "AUTHENTICATION"
    FINANCIAL = "FINANCIAL"
    CREDENTIAL = "CREDENTIAL"
    DEVICE = "DEVICE"
    MEDICAL = "MEDICAL"
    ID_NUMBER = "ID_NUMBER"


# ============================================================================
# INPUT TYPES - For mutations and complex queries
# ============================================================================

@strawberry.input
class PIIOptionsInput:
    """
    Flexible PII options - all optional so client can specify only what they want
    Defaults to True for all types if not specified
    """
    person: Optional[bool] = None
    organization: Optional[bool] = None
    location: Optional[bool] = None
    email_address: Optional[bool] = None
    phone_number: Optional[bool] = None
    credit_card: Optional[bool] = None
    ssn: Optional[bool] = None
    ip_address: Optional[bool] = None
    url: Optional[bool] = None
    date_time: Optional[bool] = None
    password: Optional[bool] = None
    api_key: Optional[bool] = None
    roll_number: Optional[bool] = None
    deploy_token: Optional[bool] = None
    authentication: Optional[bool] = None
    financial: Optional[bool] = None
    credential: Optional[bool] = None
    device: Optional[bool] = None
    medical: Optional[bool] = None
    id_number: Optional[bool] = None

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary format expected by backend"""
        mapping = {
            'person': 'PERSON',
            'organization': 'ORGANIZATION',
            'location': 'LOCATION',
            'email_address': 'EMAIL_ADDRESS',
            'phone_number': 'PHONE_NUMBER',
            'credit_card': 'CREDIT_CARD',
            'ssn': 'SSN',
            'ip_address': 'IP_ADDRESS',
            'url': 'URL',
            'date_time': 'DATE_TIME',
            'password': 'PASSWORD',
            'api_key': 'API_KEY',
            'roll_number': 'ROLL_NUMBER',
            'deploy_token': 'DEPLOY_TOKEN',
            'authentication': 'AUTHENTICATION',
            'financial': 'FINANCIAL',
            'credential': 'CREDENTIAL',
            'device': 'DEVICE',
            'medical': 'MEDICAL',
            'id_number': 'ID_NUMBER',
        }
        
        result = {}
        for field_name, backend_name in mapping.items():
            value = getattr(self, field_name, None)
            if value is not None:
                result[backend_name] = value
        
        return result if result else None


@strawberry.input
class AnonymizationInput:
    """Input for single text anonymization"""
    text: str
    options: Optional[PIIOptionsInput] = None
    full_redaction: bool = True


@strawberry.input
class BatchAnonymizationInput:
    """Input for batch anonymization"""
    texts: List[str]
    options: Optional[PIIOptionsInput] = None
    full_redaction: bool = True


@strawberry.input
class DetectionInput:
    """Input for entity detection without anonymization"""
    text: str
    options: Optional[PIIOptionsInput] = None


# ============================================================================
# OUTPUT TYPES - Response types
# ============================================================================

@strawberry.type
class Entity:
    """Detected entity information"""
    entity_group: str
    word: str
    start: int
    end: int
    score: float
    detector: str
    
    # Computed fields for additional context
    @strawberry.field
    def confidence_percentage(self) -> float:
        """Score as percentage for better UX"""
        return round(self.score * 100, 2)
    
    @strawberry.field
    def is_high_confidence(self) -> bool:
        """Whether this detection has high confidence (>80%)"""
        return self.score > 0.8


@strawberry.type
class Metadata:
    """Metadata about the anonymization process"""
    total_entities: int
    domains_used: List[str]
    detectors_used: List[str]
    
    # Additional computed fields
    @strawberry.field
    def unique_entity_types(self) -> int:
        """Count of unique entity types detected"""
        return len(set(self.detectors_used))


@strawberry.type
class AnonymizationResult:
    """Result of text anonymization"""
    anonymized_text: str
    entities: List[Entity]
    processing_time: float
    domains_detected: List[str]
    entities_processed: int
    strategy_used: AnonymizationStrategy
    metadata: Metadata
    
    # Computed fields for analytics
    @strawberry.field
    def entities_by_type(self) -> strawberry.scalars.JSON:
        """Group entities by type for analytics"""
        grouped = {}
        for entity in self.entities:
            entity_type = entity.entity_group
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append({
                'word': entity.word,
                'score': entity.score,
                'position': f"{entity.start}-{entity.end}"
            })
        return grouped
    
    @strawberry.field
    def redaction_percentage(self) -> float:
        """Percentage of text that was redacted"""
        if not self.anonymized_text:
            return 0.0
        original_length = sum(len(e.word) for e in self.entities)
        if original_length == 0:
            return 0.0
        return round((original_length / len(self.anonymized_text)) * 100, 2)


@strawberry.type
class BatchAnonymizationResult:
    """Result of batch anonymization"""
    results: List[AnonymizationResult]
    batch_size: int
    total_processing_time: float
    average_time_per_text: float
    strategy_used: AnonymizationStrategy
    
    # Computed analytics
    @strawberry.field
    def total_entities_found(self) -> int:
        """Total entities across all texts"""
        return sum(r.entities_processed for r in self.results)
    
    @strawberry.field
    def success_rate(self) -> float:
        """Percentage of successfully processed texts"""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.anonymized_text)
        return round((successful / len(self.results)) * 100, 2)


@strawberry.type
class DetectionResult:
    """Result of entity detection without anonymization"""
    entities: List[Entity]
    domains_detected: List[str]
    total_entities: int
    entity_types: List[str]
    processing_time: float
    
    # Analytics
    @strawberry.field
    def entities_by_confidence(self) -> strawberry.scalars.JSON:
        """Group entities by confidence level"""
        high = [e for e in self.entities if e.score > 0.8]
        medium = [e for e in self.entities if 0.5 < e.score <= 0.8]
        low = [e for e in self.entities if e.score <= 0.5]
        
        return {
            'high': len(high),
            'medium': len(medium),
            'low': len(low),
            'details': {
                'high': [{'word': e.word, 'type': e.entity_group} for e in high],
                'medium': [{'word': e.word, 'type': e.entity_group} for e in medium],
                'low': [{'word': e.word, 'type': e.entity_group} for e in low],
            }
        }


# ============================================================================
# HEALTH & STATUS TYPES
# ============================================================================

@strawberry.type
class ComponentHealth:
    """Health status of individual components"""
    detection_engine: str
    anonymization_engine: str
    model_manager: str
    mcp_servers: str


@strawberry.type
class MCPServerStatus:
    """Status of individual MCP server"""
    name: str
    running: bool
    healthy: bool
    port: int
    url: str
    pid: Optional[int] = None
    uptime: Optional[float] = None


@strawberry.type
class MCPServersHealth:
    """Health status of all MCP servers"""
    healthy: int
    total: int
    servers: List[MCPServerStatus]
    
    @strawberry.field
    def health_percentage(self) -> float:
        """Percentage of healthy servers"""
        if self.total == 0:
            return 0.0
        return round((self.healthy / self.total) * 100, 2)


@strawberry.type
class HealthStatus:
    """Overall system health status"""
    status: HealthStatusEnum
    timestamp: str
    components: ComponentHealth
    loaded_models: Optional[int] = None
    mcp_servers: Optional[MCPServersHealth] = None
    
    @strawberry.field
    def is_operational(self) -> bool:
        """Whether system is operational"""
        return self.status in [HealthStatusEnum.HEALTHY, HealthStatusEnum.DEGRADED]


@strawberry.type
class SystemConfig:
    """System configuration information"""
    version: str
    architecture: str
    available_models: List[str]
    supported_entities: List[str]
    anonymization_strategies: List[str]
    model_stats: strawberry.scalars.JSON
    
    @strawberry.field
    def total_pii_types(self) -> int:
        """Total number of supported PII types"""
        return len(self.supported_entities)


# ============================================================================
# ERROR TYPES - For better error handling
# ============================================================================

@strawberry.type
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    code: str


@strawberry.type
class AnonymizationError:
    """Error during anonymization"""
    message: str
    code: str
    details: Optional[strawberry.scalars.JSON] = None
    timestamp: str


# ============================================================================
# UNION TYPES - For operations that can succeed or fail
# ============================================================================

@strawberry.type
class AnonymizationSuccess:
    """Successful anonymization"""
    result: AnonymizationResult


@strawberry.type
class AnonymizationFailure:
    """Failed anonymization"""
    error: AnonymizationError


# Union type for better error handling
AnonymizationResponse = strawberry.union(
    "AnonymizationResponse",
    (AnonymizationSuccess, AnonymizationFailure)
)


# ============================================================================
# QUERY TYPE - Read operations
# ============================================================================

@strawberry.type
class Query:
    """
    GraphQL Query root
    All read-only operations
    """
    
    @strawberry.field
    async def health(self) -> HealthStatus:
        """Get system health status"""
        from graphql_resolvers import resolve_health
        return await resolve_health()
    
    @strawberry.field
    async def mcp_status(self) -> MCPServersHealth:
        """Get detailed MCP server status"""
        from graphql_resolvers import resolve_mcp_status
        return await resolve_mcp_status()
    
    @strawberry.field
    async def config(self) -> SystemConfig:
        """Get system configuration"""
        from graphql_resolvers import resolve_config
        return await resolve_config()
    
    @strawberry.field
    async def detect_entities(
        self,
        text: str,
        options: Optional[PIIOptionsInput] = None
    ) -> DetectionResult:
        """
        Detect entities without anonymization
        Useful for preview/analysis
        """
        from graphql_resolvers import resolve_detect_entities
        return await resolve_detect_entities(text, options)
    
    @strawberry.field
    async def supported_pii_types(self) -> List[PIITypeEnum]:
        """Get list of all supported PII types"""
        return list(PIITypeEnum)
    
    @strawberry.field
    async def supported_strategies(self) -> List[AnonymizationStrategy]:
        """Get list of all supported anonymization strategies"""
        return list(AnonymizationStrategy)


# ============================================================================
# MUTATION TYPE - Write operations
# ============================================================================

@strawberry.type
class Mutation:
    """
    GraphQL Mutation root
    All write/action operations
    """
    
    @strawberry.field
    async def anonymize(
        self,
        text: str,
        options: Optional[PIIOptionsInput] = None,
        full_redaction: bool = True
    ) -> AnonymizationResult:
        """
        Anonymize a single text
        
        Args:
            text: Input text to anonymize
            options: PII types to detect (all if not specified)
            full_redaction: Use full redaction vs partial masking
        
        Returns:
            AnonymizationResult with anonymized text and metadata
        
        Raises:
            ValidationError: If text is empty or invalid
        """
        from graphql_resolvers import resolve_anonymize
        return await resolve_anonymize(text, options, full_redaction)
    
    @strawberry.field
    async def anonymize_batch(
        self,
        texts: List[str],
        options: Optional[PIIOptionsInput] = None,
        full_redaction: bool = True
    ) -> BatchAnonymizationResult:
        """
        Anonymize multiple texts in batch
        More efficient than multiple single requests
        
        Args:
            texts: List of texts to anonymize
            options: PII types to detect
            full_redaction: Use full redaction vs partial masking
        
        Returns:
            BatchAnonymizationResult with results for each text
        """
        from graphql_resolvers import resolve_anonymize_batch
        return await resolve_anonymize_batch(texts, options, full_redaction)
    
    @strawberry.field
    async def anonymize_with_preview(
        self,
        text: str,
        options: Optional[PIIOptionsInput] = None
    ) -> DetectionResult:
        """
        Preview what would be anonymized without actually anonymizing
        Useful for user confirmation workflows
        """
        from graphql_resolvers import resolve_anonymize_with_preview
        return await resolve_anonymize_with_preview(text, options)


# ============================================================================
# SUBSCRIPTION TYPE - Real-time updates (future feature)
# ============================================================================

@strawberry.type
class Subscription:
    """
    GraphQL Subscription root
    Real-time updates for long-running operations
    """
    
    @strawberry.subscription
    async def anonymization_progress(
        self,
        job_id: str
    ) -> strawberry.scalars.JSON:
        """
        Subscribe to anonymization progress for large documents
        Useful for processing very large texts or batches
        
        Returns progress updates as:
        {
            "job_id": "...",
            "progress": 0.75,
            "status": "processing",
            "entities_found": 42,
            "estimated_time_remaining": 5.2
        }
        """
        # Future implementation for real-time progress
        pass


# ============================================================================
# SCHEMA - Combine all types
# ============================================================================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
