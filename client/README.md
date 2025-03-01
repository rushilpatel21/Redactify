# Redactify - Advanced PII Anonymization Platform

![Redactify](https://img.shields.io/badge/Redactify-1.0.0-blue)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB)

Redactify is a modern, user-friendly web application designed to anonymize personally identifiable information (PII) in text documents. This frontend application provides an intuitive interface for users to redact sensitive information while maintaining document utility and readability.

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [API Integration](#-api-integration)
- [Customization Options](#-customization-options)
- [UI Components](#-ui-components)
- [Animation System](#-animation-system)
  
## ✨ Features

- **Text Processing**: Input and output text areas with character count
- **Customizable PII Detection**: Selectively enable/disable specific PII types
- **Redaction Options**: Choose between full and partial redaction modes
- **Real-time Feedback**: Toast notifications for success, errors, and warnings
- **Responsive Design**: Fully responsive layout that works on all screen sizes
- **Copy Functionality**: One-click copy for anonymized text results
- **Animated Interface**: Smooth transitions and interactive elements

## 🛠 Tech Stack

- **React**: Frontend library for building the user interface
- **Framer Motion**: Animation library for creating smooth transitions and effects
- **React Icons**: Icon library providing various icon sets
- **SweetAlert2**: Library for beautiful and customizable toast notifications

### Package Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "framer-motion": "^10.12.16",
    "react-icons": "^4.10.1",
    "sweetalert2": "^11.7.12"
  }
}
```

## 🚀 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/rushilpatel21/Redactify.git
cd redactify
```

2. **Install dependencies**

```bash
npm install
```

3. **Create environment file**

Create a `.env` file in the root directory with the following content:

```
VITE_BACKEND_BASE_URL='http://localhost:8000'
```

4. **Run development server**

```bash
npm run dev
```

5. **Build for production**

```bash
npm run build
```

## 🔧 Environment Configuration

The application uses environment variables to configure the backend API URL. Create a `.env` file in the project root with the following configuration:

```
VITE_BACKEND_BASE_URL='http://localhost:8000'
```

For production deployment, update this URL to your production API endpoint.

## 🔌 API Integration

### Endpoint

The application interacts with a single endpoint:

```
POST /anonymize
```

This endpoint is accessed using the `BASE_URL` defined in the environment variables.

### Request Format

```javascript
{
  "text": "The input text containing PII to anonymize",
  "options": {
    "PERSON": true,
    "ORGANIZATION": true,
    "LOCATION": true,
    "EMAIL_ADDRESS": true,
    "PHONE_NUMBER": true,
    "CREDIT_CARD": true,
    "SSN": true,
    "IP_ADDRESS": true,
    "URL": true,
    "DATE_TIME": true,
    "PASSWORD": true,
    "API_KEY": true,
    "ROLL_NUMBER": true
  },
  "full_redaction": true
}
```

### Response Format

Successful response:

```javascript
{
  "anonymized_text": "The redacted text with PII anonymized"
}
```

Error response:

```javascript
{
  "error": "Description of the error that occurred"
}
```

## 🎛 Customization Options

### PII Types

Redactify supports the following PII types for anonymization:

| PII Type | Description | Icon | 
|----------|-------------|------|
| PERSON | Names and personal identifiers | Person icon |
| ORGANIZATION | Company names, organizations | Building icon |
| LOCATION | Addresses, cities, countries | Location pin icon |
| EMAIL_ADDRESS | Email addresses | Email icon |
| PHONE_NUMBER | Phone and fax numbers | Phone icon |
| CREDIT_CARD | Credit card numbers | Credit card icon |
| SSN | Social security numbers | Security icon |
| IP_ADDRESS | IP addresses | Network icon |
| URL | Web URLs | World wide web icon |
| DATE_TIME | Date and time expressions | Calendar icon |
| PASSWORD | Password strings | Password icon |
| API_KEY | API keys and authentication tokens | Key icon |
| ROLL_NUMBER | Student IDs, roll numbers | Code icon |

### Redaction Modes

- **Full Redaction**: Completely replaces the detected PII with a generic placeholder (e.g., [PERSON])
- **Partial Redaction**: Preserves some characters in the detected PII for context while masking others (e.g., J*** S***)

### Selection Controls

- **Select All**: Enables all PII types for detection
- **Clear All**: Disables all PII types for detection
- **Individual Toggles**: Enable/disable specific PII types as needed

## 🎨 UI Components

### Text Processing Area

- **Input Text Box**: Accepts text that requires anonymization
- **Output Text Box**: Displays the anonymized result
- **Character Count**: Shows the length of input and output text
- **Copy Button**: Copies anonymized text to clipboard
- **Clear Button**: Resets input and output fields

### Options Panel

- **Redaction Mode Selector**: Toggle between full and partial redaction
- **PII Type Grid**: Visual selection of PII types to detect
- **Action Buttons**: Select All and Clear All options

### Notifications

- **Toast Notifications**: Non-intrusive alerts at the bottom of the screen
- **Loading Indicator**: Visual feedback during anonymization processing
- **Success/Error Feedback**: Contextual messages based on operation outcome

## ✨ Animation System

The application uses Framer Motion for animations:

- **Entrance Animations**: Smooth fade and slide effects when components mount
- **Interactive Animations**: Subtle hover and tap effects on interactive elements
- **Staggered Animations**: PII type options appear with a slight delay between each item
- **Loading Animation**: Rotating spinner when processing text


---

&copy; 2025 Redactify. All rights reserved.
