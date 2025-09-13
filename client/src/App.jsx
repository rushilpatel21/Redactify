import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaShieldAlt, FaCheck, FaStar, FaKey, FaFingerprint, FaMedkit } from 'react-icons/fa';
import { HiCode, HiLockClosed } from 'react-icons/hi';
import { MdSecurity, MdPersonOutline, MdOutlineEmail, MdPhone, MdDevices } from 'react-icons/md';
import { BsBuilding, BsGeoAlt, BsCreditCard, BsCalendarDate, BsKey, BsStars } from 'react-icons/bs';
import { CgPassword } from 'react-icons/cg';
import { TbWorldWww, TbNetwork, TbId } from 'react-icons/tb';
import { RiMoneyDollarCircleLine } from 'react-icons/ri';
import Swal from 'sweetalert2';
import './App.css';

const BASE_URL = import.meta.env.VITE_BACKEND_BASE_URL || "";

const piiTypes = [
  { id: "PERSON", icon: <MdPersonOutline /> },
  { id: "ORGANIZATION", icon: <BsBuilding /> },
  { id: "LOCATION", icon: <BsGeoAlt /> },
  { id: "EMAIL_ADDRESS", icon: <MdOutlineEmail /> },
  { id: "PHONE_NUMBER", icon: <MdPhone /> },
  { id: "CREDIT_CARD", icon: <BsCreditCard /> },
  { id: "SSN", icon: <MdSecurity /> },
  { id: "IP_ADDRESS", icon: <TbNetwork /> },
  { id: "URL", icon: <TbWorldWww /> },
  { id: "DATE_TIME", icon: <BsCalendarDate /> },
  { id: "PASSWORD", icon: <CgPassword /> },
  { id: "API_KEY", icon: <BsKey /> },
  { id: "DEPLOY_TOKEN", icon: <FaKey /> },
  { id: "AUTHENTICATION", icon: <HiLockClosed /> },
  { id: "FINANCIAL", icon: <RiMoneyDollarCircleLine /> },
  { id: "CREDENTIAL", icon: <FaFingerprint /> },
  { id: "ROLL_NUMBER", icon: <HiCode /> },
  { id: "DEVICE", icon: <MdDevices /> },
  { id: "MEDICAL", icon: <FaMedkit /> },
  { id: "ID_NUMBER", icon: <TbId /> }
];

// Toast configuration for SweetAlert2
const Toast = Swal.mixin({
  toast: true,
  position: 'bottom',
  showConfirmButton: false,
  timer: 3000,
  timerProgressBar: true,
  didOpen: (toast) => {
    toast.addEventListener('mouseenter', Swal.stopTimer)
    toast.addEventListener('mouseleave', Swal.resumeTimer)
  },
  customClass: {
    popup: 'swal-toast-popup neubrutal-toast',
    title: 'swal-toast-title',
    icon: 'swal-toast-icon'
  }
});

function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [userRequest, setUserRequest] = useState("Please analyze this text and intelligently detect all personally identifiable information (PII) including names, organizations, locations, contact details, financial information, medical data, technical credentials, and any other sensitive information. Apply comprehensive anonymization using the most appropriate detection models and redaction strategies to ensure complete privacy protection while maintaining text readability and context.");
  const [options, setOptions] = useState(
    piiTypes.reduce((acc, type) => ({ ...acc, [type.id]: true }), {})
  );
  const [fullRedaction, setFullRedaction] = useState(true);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [useTrueMCP, setUseTrueMCP] = useState(true);
  const [executionPlan, setExecutionPlan] = useState(null);

  const handleOptionChange = (e) => {
    const { name, checked } = e.target;
    setOptions((prev) => ({ ...prev, [name]: checked }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!inputText.trim()) {
      Toast.fire({
        icon: 'warning',
        title: 'Please enter some text to anonymize',
      });
      return;
    }
    
    setLoading(true);
    setExecutionPlan(null);
    
    try {
      // Always use LLM endpoint
      const endpoint = '/anonymize_llm';
      
      // Prepare request body
      const requestBody = {
        user_request: userRequest,
        text: inputText,
        options: options
      };
      
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      });
      
      const data = await response.json();
      
      if (data.anonymized_text) {
        setOutputText(data.anonymized_text);
        
        // Store execution plan
        if (data.execution_plan) {
          setExecutionPlan(data.execution_plan);
        }
        
        const processingTime = data.total_processing_time || data.processing_time || 0;
        const method = data.orchestration_method || 'standard';
        
        Toast.fire({
          icon: 'success',
          title: `Text anonymized successfully (${processingTime.toFixed(2)}s, ${method})`,
          timer: 4000
        });
      } else {
        setOutputText("Error: " + (data.error || 'Unknown error'));
        Toast.fire({
          icon: 'error',
          title: data.error || 'Something went wrong',
        });
      }
    } catch (err) {
      setOutputText("An error occurred: " + err.message);
      Toast.fire({
        icon: 'error',
        title: err.message,
      });
    }
    setLoading(false);
  };

  const copyToClipboard = () => {
    if (!outputText) return;
    
    navigator.clipboard.writeText(outputText).then(() => {
      setCopied(true);
      Toast.fire({
        icon: 'success',
        title: 'Copied to clipboard',
        timer: 1500,
      });
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const selectAllOptions = () => {
    const newOptions = {};
    piiTypes.forEach(type => {
      newOptions[type.id] = true;
    });
    setOptions(newOptions);
  };

  const clearAllOptions = () => {
    const newOptions = {};
    piiTypes.forEach(type => {
      newOptions[type.id] = false;
    });
    setOptions(newOptions);
  };

  const clearForm = () => {
    setInputText("");
    setOutputText("");
    setExecutionPlan(null);
    setUserRequest("Please analyze this text and intelligently detect all personally identifiable information (PII) including names, organizations, locations, contact details, financial information, medical data, technical credentials, and any other sensitive information. Apply comprehensive anonymization using the most appropriate detection models and redaction strategies to ensure complete privacy protection while maintaining text readability and context.");
  };

  return (
    <div className="app-container">
      <div className="decorative-star star-1"><FaStar /></div>
      <div className="decorative-star star-2"><BsStars /></div>
      <div className="decorative-star star-3"><FaStar /></div>
      <div className="decorative-star star-4"><BsStars /></div>
      
      <motion.header 
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="header-content">
          <div className="logo-container">
            <FaShieldAlt className="logo-icon" />
            <h1>Redactify</h1>
          </div>
          <p>Advanced PII Anonymization Platform</p>
        </div>
      </motion.header>
      
      <main>
        <motion.div 
          className="card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <form onSubmit={handleSubmit}>
            <section className="text-sections">
              <div className="text-box">
                <div className="text-box-header">
                  <label htmlFor="inputText">Input Text</label>
                  <button 
                    type="button" 
                    className="clear-button"
                    onClick={clearForm}
                  >
                    Clear All
                  </button>
                </div>
                <textarea
                  id="inputText"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste your text here to detect and anonymize personally identifiable information..."
                  className="text-area input-area"
                />
                <div className="text-meta">
                  <span>{inputText.length} characters</span>
                </div>
              </div>
              
              <div className="text-box">
                <div className="text-box-header">
                  <label htmlFor="outputText">Anonymized Output</label>
                  <button 
                    type="button" 
                    className={`copy-button ${copied ? 'copied' : ''}`}
                    onClick={copyToClipboard}
                    disabled={!outputText}
                  >
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <textarea
                  id="outputText"
                  value={outputText}
                  readOnly
                  placeholder="Anonymized text will appear here..."
                  className="text-area output-area"
                />
                <div className="text-meta">
                  <span>{outputText.length} characters</span>
                </div>
              </div>

              {/* LLM Execution Plan */}
              {executionPlan && (
                <div className="execution-plan-box">
                  <div className="text-box-header">
                    <label>LLM Execution Plan</label>
                  </div>
                  <div className="execution-plan-content">
                    <div className="plan-tools">
                      <strong>Tools Selected:</strong>
                      <ul>
                        {executionPlan.tool_calls?.map((tool, index) => (
                          <li key={index}>
                            <strong>{tool.tool}:</strong> {tool.reasoning}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </section>

            <motion.section 
              className="options-section"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.5 }}
            >
              <h3 className="section-title">
                PII Detection Options
              </h3>
              
              <div className="pii-options">
                <div className="pii-header">
                  <h3>Select PII Types to Anonymize:</h3>
                  <div className="pii-actions">
                    <button type="button" onClick={selectAllOptions} className="action-button">Select All</button>
                    <button type="button" onClick={clearAllOptions} className="action-button">Clear All</button>
                  </div>
                </div>
                
                <div className="pii-grid">
                  <AnimatePresence>
                    {piiTypes.map((type) => (
                      <motion.label 
                        key={type.id} 
                        className={`pii-option ${options[type.id] ? 'selected' : ''}`}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.2 }}
                      >
                        <input
                          type="checkbox"
                          name={type.id}
                          checked={options[type.id]}
                          onChange={handleOptionChange}
                        />
                        <div className="pii-icon">{type.icon}</div>
                        <div className="pii-label">{type.id.replace(/_/g, ' ')}</div>
                        {options[type.id] && (
                          <FaCheck className="check-icon" />
                        )}
                      </motion.label>
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            </motion.section>

            <motion.button 
              type="submit" 
              className="submit-button"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.5 }}
            >
              {loading ? (
                <div className="loading-spinner">
                  <div className="spinner"></div>
                  <span>Processing...</span>
                </div>
              ) : "Anonymize Text"}
            </motion.button>
          </form>
        </motion.div>

        <motion.div 
          className="info-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          <h3>Why Anonymize PII?</h3>
          <p>
            Protecting Personally Identifiable Information (PII) is crucial for compliance with data 
            privacy regulations like GDPR, CCPA, and HIPAA. Redactify helps safeguard sensitive 
            information while maintaining data utility for analysis and processing.
          </p>
        </motion.div>
      </main>
      
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.5 }}
      >
        <div className="footer-content">
          <p>&copy; 2025 Redactify. All rights reserved.</p>
          <p>Secure • Reliable • Compliant</p>
        </div>
      </motion.footer>
    </div>
  );
}

export default App;