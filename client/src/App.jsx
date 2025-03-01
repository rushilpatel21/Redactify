import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaShieldAlt, FaCheck } from 'react-icons/fa';
import { HiCode } from 'react-icons/hi';
import { MdSecurity, MdPersonOutline, MdOutlineEmail, MdPhone } from 'react-icons/md';
import { BsBuilding, BsGeoAlt, BsCreditCard, BsCalendarDate, BsKey } from 'react-icons/bs';
import { CgPassword } from 'react-icons/cg';
import { TbWorldWww, TbNetwork } from 'react-icons/tb';
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
  { id: "ROLL_NUMBER", icon: <HiCode /> }
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
    popup: 'swal-toast-popup',
    title: 'swal-toast-title',
    icon: 'swal-toast-icon'
  }
});

function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [options, setOptions] = useState(
    piiTypes.reduce((acc, type) => ({ ...acc, [type.id]: true }), {})
  );
  const [fullRedaction, setFullRedaction] = useState(true);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

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
    try {
      const response = await fetch(`${BASE_URL}/anonymize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: inputText,
          options: options,
          full_redaction: fullRedaction
        })
      });
      const data = await response.json();
      if (data.anonymized_text) {
        setOutputText(data.anonymized_text);
        Toast.fire({
          icon: 'success',
          title: 'Text anonymized successfully',
        });
      } else {
        setOutputText("Error: " + data.error);
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
  };

  return (
    <div className="app-container">
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
            </section>

            <motion.section 
              className="options-section"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.5 }}
            >
              <h3 className="section-title">Redaction Options</h3>
              
              <div className="redaction-toggle">
                <label className={`option-pill ${fullRedaction ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="redactionMode"
                    value="full"
                    checked={fullRedaction === true}
                    onChange={() => setFullRedaction(true)}
                  />
                  <span>Full Redaction</span>
                </label>
                <label className={`option-pill ${!fullRedaction ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="redactionMode"
                    value="partial"
                    checked={fullRedaction === false}
                    onChange={() => setFullRedaction(false)}
                  />
                  <span>Partial Redaction</span>
                </label>
              </div>
              
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