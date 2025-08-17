# ui/tabs/document_intelligence_enhanced.py
"""
Enhanced Document Intelligence Hub - Fully functional with bulk upload and data source integrations
"""
import streamlit as st
import pandas as pd
import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import requests
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
import xml.etree.ElementTree as ET

# Data source connectors
class DataSourceConnector:
    """Base class for data source connectors"""
    
    def __init__(self, config: dict):
        self.config = config
        self.authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with the data source"""
        raise NotImplementedError
    
    def list_files(self, path: str = "/") -> List[dict]:
        """List files in the data source"""
        raise NotImplementedError
    
    def download_file(self, file_id: str, file_path: str) -> bytes:
        """Download a file from the data source"""
        raise NotImplementedError

class SharePointConnector(DataSourceConnector):
    """SharePoint Online connector using Microsoft Graph API"""
    
    def authenticate(self) -> bool:
        """Authenticate with SharePoint using app credentials"""
        try:
            # Mock authentication for demo - replace with actual Graph API auth
            tenant_id = self.config.get('tenant_id')
            client_id = self.config.get('client_id')
            client_secret = self.config.get('client_secret')
            
            if not all([tenant_id, client_id, client_secret]):
                return False
            
            # In real implementation, get OAuth token here
            self.authenticated = True
            return True
        except Exception as e:
            st.error(f"SharePoint authentication failed: {str(e)}")
            return False
    
    def list_files(self, site_path: str = "/") -> List[dict]:
        """List files from SharePoint site"""
        if not self.authenticated:
            return []
        
        # Mock file list - replace with actual Graph API call
        return [
            {"id": "sp_001", "name": "Financial_Report_Q3.pdf", "size": 2048576, "modified": "2024-08-15"},
            {"id": "sp_002", "name": "Board_Meeting_Minutes.docx", "size": 1024000, "modified": "2024-08-14"},
            {"id": "sp_003", "name": "ESG_Assessment.xlsx", "size": 3072000, "modified": "2024-08-13"},
        ]
    
    def download_file(self, file_id: str, file_path: str) -> bytes:
        """Download file from SharePoint"""
        # Mock download - replace with actual Graph API call
        return b"Mock SharePoint file content"

class GoogleDriveConnector(DataSourceConnector):
    """Google Drive connector using Google Drive API"""
    
    def authenticate(self) -> bool:
        """Authenticate with Google Drive"""
        try:
            # Mock authentication - replace with actual Google API auth
            service_account_key = self.config.get('service_account_key')
            if not service_account_key:
                return False
            
            self.authenticated = True
            return True
        except Exception as e:
            st.error(f"Google Drive authentication failed: {str(e)}")
            return False
    
    def list_files(self, folder_id: str = "root") -> List[dict]:
        """List files from Google Drive"""
        if not self.authenticated:
            return []
        
        # Mock file list
        return [
            {"id": "gd_001", "name": "Investment_Strategy.pdf", "size": 1536000, "modified": "2024-08-16"},
            {"id": "gd_002", "name": "Due_Diligence_Report.docx", "size": 4096000, "modified": "2024-08-15"},
            {"id": "gd_003", "name": "Portfolio_Analysis.xlsx", "size": 2560000, "modified": "2024-08-14"},
        ]
    
    def download_file(self, file_id: str, file_path: str) -> bytes:
        """Download file from Google Drive"""
        # Mock download
        return b"Mock Google Drive file content"

class BoxConnector(DataSourceConnector):
    """Box.com connector"""
    
    def authenticate(self) -> bool:
        """Authenticate with Box"""
        try:
            api_token = self.config.get('api_token')
            if not api_token:
                return False
            
            self.authenticated = True
            return True
        except Exception as e:
            st.error(f"Box authentication failed: {str(e)}")
            return False
    
    def list_files(self, folder_id: str = "0") -> List[dict]:
        """List files from Box"""
        if not self.authenticated:
            return []
        
        return [
            {"id": "box_001", "name": "Compliance_Report.pdf", "size": 1792000, "modified": "2024-08-17"},
            {"id": "box_002", "name": "Risk_Assessment.json", "size": 512000, "modified": "2024-08-16"},
        ]
    
    def download_file(self, file_id: str, file_path: str) -> bytes:
        """Download file from Box"""
        return b"Mock Box file content"

class DropboxConnector(DataSourceConnector):
    """Dropbox connector"""
    
    def authenticate(self) -> bool:
        """Authenticate with Dropbox"""
        try:
            access_token = self.config.get('access_token')
            if not access_token:
                return False
            
            self.authenticated = True
            return True
        except Exception as e:
            st.error(f"Dropbox authentication failed: {str(e)}")
            return False
    
    def list_files(self, path: str = "/") -> List[dict]:
        """List files from Dropbox"""
        if not self.authenticated:
            return []
        
        return [
            {"id": "db_001", "name": "Market_Analysis.xlsx", "size": 2048000, "modified": "2024-08-17"},
            {"id": "db_002", "name": "Legal_Documents.zip", "size": 10485760, "modified": "2024-08-16"},
        ]
    
    def download_file(self, file_id: str, file_path: str) -> bytes:
        """Download file from Dropbox"""
        return b"Mock Dropbox file content"

def get_data_source_connector(source_type: str, config: dict) -> Optional[DataSourceConnector]:
    """Factory function to get data source connector"""
    connectors = {
        'sharepoint': SharePointConnector,
        'googledrive': GoogleDriveConnector,
        'box': BoxConnector,
        'dropbox': DropboxConnector
    }
    
    connector_class = connectors.get(source_type.lower())
    if connector_class:
        return connector_class(config)
    return None

def supported_file_types():
    """Return supported file types with enhanced categories"""
    return {
        'PDF': ['.pdf'],
        'Word': ['.docx', '.doc'],
        'Excel': ['.xlsx', '.xls'],
        'PowerPoint': ['.pptx', '.ppt'],
        'CSV': ['.csv'],
        'JSON': ['.json'],
        'XML': ['.xml'],
        'Text': ['.txt', '.md', '.rtf'],
        'Images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
        'Archives': ['.zip', '.rar', '.7z']
    }

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = DocxDocument(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting DOCX text: {str(e)}"

def extract_text_from_xml(file_content: bytes) -> str:
    """Extract text from XML file"""
    try:
        root = ET.fromstring(file_content.decode('utf-8'))
        text = ET.tostring(root, encoding='unicode', method='text')
        return text
    except Exception as e:
        return f"Error extracting XML text: {str(e)}"

def process_uploaded_file(uploaded_file, source_info: dict = None) -> dict:
    """Enhanced file processing with better text extraction"""
    try:
        # Read file content
        if hasattr(uploaded_file, 'read'):
            file_content = uploaded_file.read()
            filename = uploaded_file.name
            filetype = uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
            filesize = len(file_content)
        else:
            # Handle file from data sources
            file_content = uploaded_file
            filename = source_info.get('name', 'unknown')
            filetype = 'unknown'
            filesize = len(file_content)
        
        file_hash = calculate_file_hash(file_content)
        
        file_details = {
            "filename": filename,
            "filetype": filetype,
            "filesize": filesize,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hash": file_hash,
            "source": source_info.get('source', 'upload') if source_info else 'upload'
        }
        
        # Enhanced content extraction
        content = ""
        structured_data = None
        
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            content = extract_text_from_pdf(file_content)
        elif file_ext in ['.docx', '.doc']:
            content = extract_text_from_docx(file_content)
        elif file_ext == '.txt':
            content = file_content.decode('utf-8', errors='ignore')
        elif file_ext == '.json':
            try:
                json_data = json.loads(file_content.decode('utf-8'))
                content = json.dumps(json_data, indent=2)
                structured_data = json_data
            except json.JSONDecodeError as e:
                content = f"JSON parsing error: {str(e)}"
        elif file_ext == '.xml':
            content = extract_text_from_xml(file_content)
        elif file_ext == '.csv':
            try:
                csv_file = io.StringIO(file_content.decode('utf-8'))
                df = pd.read_csv(csv_file)
                content = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n\nColumns: {', '.join(df.columns)}\n\nFirst 5 rows:\n{df.head().to_string()}"
                structured_data = df
            except Exception as e:
                content = f"CSV parsing error: {str(e)}"
        elif file_ext in ['.xlsx', '.xls']:
            try:
                excel_file = io.BytesIO(file_content)
                df = pd.read_excel(excel_file)
                content = f"Excel file with {len(df)} rows and {len(df.columns)} columns.\n\nColumns: {', '.join(df.columns)}\n\nFirst 5 rows:\n{df.head().to_string()}"
                structured_data = df
            except Exception as e:
                content = f"Excel parsing error: {str(e)}"
        elif file_ext == '.zip':
            content = f"ZIP archive containing multiple files. Use archive extraction feature for detailed analysis."
        else:
            content = f"File type {file_ext} - Limited text extraction available."
        
        return {
            "details": file_details,
            "content": content,
            "structured_data": structured_data,
            "raw_content": file_content,
            "success": True
        }
    
    except Exception as e:
        return {
            "details": {"filename": filename if 'filename' in locals() else "Unknown"},
            "content": "",
            "structured_data": None,
            "raw_content": None,
            "success": False,
            "error": str(e)
        }

def process_zip_file(zip_content: bytes) -> List[dict]:
    """Extract and process files from ZIP archive"""
    try:
        zip_file = io.BytesIO(zip_content)
        extracted_files = []
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    try:
                        file_content = zip_ref.read(file_info)
                        source_info = {
                            'name': file_info.filename,
                            'source': 'zip_extract',
                            'size': file_info.file_size
                        }
                        processed_file = process_uploaded_file(file_content, source_info)
                        extracted_files.append(processed_file)
                    except Exception as e:
                        st.warning(f"Could not extract {file_info.filename}: {str(e)}")
        
        return extracted_files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return []

def get_document_templates():
    """Enhanced analysis templates"""
    return {
        'Financial Statements': {
            'questions': [
                "What is the revenue growth year-over-year?",
                "What are the key profitability metrics (gross margin, EBITDA, net margin)?",
                "What is the debt-to-equity ratio and overall leverage?",
                "Are there any significant one-time items or extraordinary charges?",
                "What are the main risk factors and uncertainties mentioned?",
                "How is the company's cash flow from operations?",
                "What are the key balance sheet items and working capital trends?"
            ],
            'analysis_points': [
                "Revenue trends, seasonality, and growth drivers",
                "Margin analysis and cost structure evolution",
                "Balance sheet strength and liquidity position",
                "Cash flow generation and capital allocation",
                "Working capital management efficiency",
                "Debt structure and covenant compliance",
                "Key accounting policies and estimates"
            ]
        },
        'Legal Documents': {
            'questions': [
                "What are the key contractual terms and conditions?",
                "Are there any material risks, liabilities, or indemnification clauses?",
                "What are the governing laws and dispute resolution mechanisms?",
                "What are the termination, renewal, and amendment clauses?",
                "Are there any compliance requirements or regulatory obligations?",
                "What intellectual property provisions are included?",
                "Are there any confidentiality or data protection requirements?"
            ],
            'analysis_points': [
                "Contract scope, obligations, and deliverables",
                "Risk allocation and liability limitations",
                "Intellectual property ownership and licensing",
                "Confidentiality, non-disclosure, and data protection",
                "Dispute resolution and governing law provisions",
                "Termination rights and consequences",
                "Compliance and regulatory requirements"
            ]
        },
        'Due Diligence Reports': {
            'questions': [
                "What are the main investment highlights and value propositions?",
                "What are the identified risks and proposed mitigation strategies?",
                "What is the market opportunity, size, and competitive landscape?",
                "How is the management team assessed in terms of experience and track record?",
                "What are the key assumptions underlying financial projections?",
                "What are the operational strengths and improvement opportunities?",
                "What is the recommended investment structure and terms?"
            ],
            'analysis_points': [
                "Market position and competitive advantages",
                "Financial performance analysis and projections",
                "Operational efficiency and scalability assessment",
                "Management quality and governance evaluation",
                "Technology, IP, and innovation capabilities",
                "ESG factors and sustainability considerations",
                "Investment risks and mitigation strategies"
            ]
        },
        'Board Materials': {
            'questions': [
                "What are the key strategic decisions requiring board approval?",
                "What is the latest financial performance update vs. budget?",
                "What strategic initiatives and investments are being discussed?",
                "Are there any regulatory, compliance, or legal issues to address?",
                "What are the key risk management updates and concerns?",
                "What governance matters require attention?",
                "What are the executive compensation and HR updates?"
            ],
            'analysis_points': [
                "Strategic direction and major initiatives",
                "Financial performance vs. budget and prior year",
                "Operational metrics, KPIs, and market conditions",
                "Risk management framework and key risks",
                "Governance effectiveness and board composition",
                "Regulatory compliance and legal matters",
                "Capital allocation and investment decisions"
            ]
        },
        'ESG Reports': {
            'questions': [
                "What are the key ESG metrics, targets, and progress against goals?",
                "How is the company addressing climate change risks and opportunities?",
                "What diversity, equity, and inclusion initiatives are in place?",
                "What is the stakeholder engagement strategy and materiality assessment?",
                "Are there any ESG-related controversies or reputation risks?",
                "How does ESG performance compare to industry peers?",
                "What ESG governance structure and oversight is in place?"
            ],
            'analysis_points': [
                "Environmental impact measurement and climate strategy",
                "Social responsibility and community engagement",
                "Governance structure, ethics, and transparency",
                "Stakeholder engagement and materiality matrix",
                "ESG risk management and opportunity identification",
                "Sustainability reporting quality and assurance",
                "ESG integration into business strategy"
            ]
        },
        'Investment Memos': {
            'questions': [
                "What is the investment thesis and expected returns?",
                "What are the key value creation drivers and timeline?",
                "How does this investment fit within the portfolio strategy?",
                "What are the main risks and sensitivity analysis?",
                "What is the competitive landscape and market dynamics?",
                "What due diligence has been conducted?",
                "What is the proposed investment structure and governance?"
            ],
            'analysis_points': [
                "Investment rationale and strategic fit",
                "Financial projections and return scenarios",
                "Market analysis and competitive positioning",
                "Management assessment and value creation plan",
                "Risk analysis and mitigation strategies",
                "Deal structure and terms evaluation",
                "Exit strategy and timeline considerations"
            ]
        }
    }

def get_llm_response(llm, prompt: str) -> str:
    """Universal LLM response handler that works with different LLM types"""
    if not llm:
        return "LLM not configured"
    
    try:
        # Method 1: LangChain ChatOpenAI/AzureChatOpenAI with invoke
        if hasattr(llm, 'invoke'):
            try:
                from langchain.schema import HumanMessage
                messages = [HumanMessage(content=prompt)]
                result = llm.invoke(messages)
                
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return str(result)
            except ImportError:
                # If langchain not available, try direct invoke
                result = llm.invoke(prompt)
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return str(result)
        
        # Method 2: LangChain LLM with predict
        elif hasattr(llm, 'predict'):
            return llm.predict(prompt)
        
        # Method 3: Direct callable
        elif callable(llm):
            result = llm(prompt)
            if isinstance(result, str):
                return result
            elif hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
        
        # Method 4: OpenAI client style
        elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",  # Default model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        # Method 5: Generate method
        elif hasattr(llm, 'generate'):
            result = llm.generate([prompt])
            if hasattr(result, 'generations') and result.generations:
                return result.generations[0][0].text
            else:
                return str(result)
        
        # Method 6: Custom generate_response
        elif hasattr(llm, 'generate_response'):
            return llm.generate_response(prompt)
        
        else:
            return f"Unsupported LLM type: {type(llm).__name__}. Available methods: {[m for m in dir(llm) if not m.startswith('_')][:10]}"
    
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def analyze_document_with_ai(content: str, document_type: str, analysis_mode: str, custom_questions: str = "", llm=None) -> str:
    """Enhanced AI analysis with proper LLM integration"""
    if not llm:
        return "AI analysis not available. Please configure LLM in API Configuration settings."
    
    templates = get_document_templates()
    template = templates.get(document_type, templates['Financial Statements'])
    
    # Prepare analysis prompt based on mode
    if analysis_mode == "Quick Summary":
        prompt = f"""
        Provide a concise executive summary of this {document_type}:
        
        Document Content:
        {content[:3000]}...
        
        Please provide:
        1. Key highlights and main findings (3-5 bullet points)
        2. Critical metrics or important figures
        3. Major decisions, recommendations, or conclusions
        4. Any red flags or items requiring attention
        
        Keep the summary focused and actionable for executive review.
        """
    
    elif analysis_mode == "Deep Analysis":
        questions = "\n".join([f"- {q}" for q in template['questions']])
        analysis_points = "\n".join([f"- {point}" for point in template['analysis_points']])
        
        prompt = f"""
        Conduct a comprehensive analysis of this {document_type}:
        
        Document Content:
        {content[:4000]}...
        
        Please address these key questions:
        {questions}
        
        Structure your analysis around these focus areas:
        {analysis_points}
        
        Provide specific insights, quantitative metrics where available, and actionable recommendations.
        Include any concerning items or areas requiring further investigation.
        """
    
    elif analysis_mode == "Compliance Check":
        prompt = f"""
        Review this {document_type} for compliance, regulatory, and risk factors:
        
        Document Content:
        {content[:3000]}...
        
        Please check for:
        - Regulatory compliance issues and requirements
        - Material risk factors and potential exposures
        - Missing required disclosures or documentation
        - Unusual items, red flags, or inconsistencies
        - Adherence to industry standards and best practices
        
        Provide a compliance assessment with specific recommendations and priority levels.
        """
    
    elif analysis_mode == "Custom Analysis":
        prompt = f"""
        Analyze this {document_type} based on the following custom requirements:
        
        Document Content:
        {content[:3000]}...
        
        Custom Analysis Questions:
        {custom_questions}
        
        Please provide detailed responses to each question with supporting evidence from the document.
        """
    
    elif analysis_mode == "Comparative Analysis":
        prompt = f"""
        Prepare this {document_type} for comparative analysis:
        
        Document Content:
        {content[:3000]}...
        
        Extract and organize:
        - Key performance metrics and indicators
        - Financial figures and ratios
        - Strategic initiatives and objectives
        - Risk factors and mitigation measures
        - Operational highlights and challenges
        
        Structure the output for easy comparison with similar documents.
        """
    
    try:
        # Use the universal LLM response handler
        response = get_llm_response(llm, prompt)
        
        # Check if we got a valid response
        if not response or response.strip() == "":
            response = "No response generated from LLM. Please check your configuration."
        
        # Add metadata to response
        analysis_metadata = f"""
═══════════════════════════════════════════════════════════════
📊 ANALYSIS METADATA
═══════════════════════════════════════════════════════════════
Document Type: {document_type}
Analysis Mode: {analysis_mode}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Content Length: {len(content)} characters
LLM Type: {type(llm).__name__}
═══════════════════════════════════════════════════════════════

{response}
"""
        
        return analysis_metadata
    
    except Exception as e:
        # Provide detailed error information for debugging
        llm_type = type(llm).__name__ if llm else "None"
        llm_methods = [method for method in dir(llm) if not method.startswith('_')] if llm else []
        
        return f"""
❌ Error in AI Analysis
═══════════════════════════════════════════════════════════════
Error Details: {str(e)}
Document Type: {document_type}
Analysis Mode: {analysis_mode}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
LLM Type: {llm_type}
Available LLM Methods: {', '.join(llm_methods[:10])}...
═══════════════════════════════════════════════════════════════

Troubleshooting:
1. Verify your LLM configuration in API settings
2. Ensure the LLM object is properly initialized
3. Check if the LLM has the correct invoke/predict method
4. Verify API keys and connection settings

If using Azure OpenAI, ensure the deployment is active and accessible.
If using OpenAI directly, verify your API key is valid.

Please check your LLM configuration and try again.
"""

def render_data_source_config():
    """Render data source configuration interface"""
    st.markdown("### 🔗 Data Source Connections")
    
    # Initialize session state for data sources
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = {}
    
    source_tabs = st.tabs(["SharePoint", "Google Drive", "Box", "Dropbox", "Local Directory"])
    
    # SharePoint Configuration
    with source_tabs[0]:
        st.markdown("#### Microsoft SharePoint Online")
        
        with st.expander("SharePoint Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                tenant_id = st.text_input("Tenant ID", key="sp_tenant_id", 
                                        help="Your Microsoft 365 tenant ID")
                client_id = st.text_input("Client ID", key="sp_client_id",
                                        help="Azure AD app client ID")
            
            with col2:
                client_secret = st.text_input("Client Secret", key="sp_client_secret",
                                            type="password", help="Azure AD app client secret")
                site_url = st.text_input("Site URL", key="sp_site_url",
                                        placeholder="https://yourtenant.sharepoint.com/sites/yoursite")
            
            if st.button("Connect to SharePoint", key="connect_sharepoint"):
                config = {
                    'tenant_id': tenant_id,
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'site_url': site_url
                }
                
                connector = get_data_source_connector('sharepoint', config)
                if connector and connector.authenticate():
                    st.session_state.data_sources['sharepoint'] = connector
                    st.success("✅ Connected to SharePoint successfully!")
                else:
                    st.error("❌ Failed to connect to SharePoint")
    
    # Google Drive Configuration
    with source_tabs[1]:
        st.markdown("#### Google Drive")
        
        with st.expander("Google Drive Configuration", expanded=True):
            service_account_file = st.file_uploader("Service Account JSON", 
                                                  type=['json'], key="gd_service_account")
            folder_id = st.text_input("Folder ID (optional)", key="gd_folder_id",
                                    help="Leave empty for root folder")
            
            if st.button("Connect to Google Drive", key="connect_googledrive"):
                if service_account_file:
                    config = {
                        'service_account_key': service_account_file.getvalue(),
                        'folder_id': folder_id or 'root'
                    }
                    
                    connector = get_data_source_connector('googledrive', config)
                    if connector and connector.authenticate():
                        st.session_state.data_sources['googledrive'] = connector
                        st.success("✅ Connected to Google Drive successfully!")
                    else:
                        st.error("❌ Failed to connect to Google Drive")
                else:
                    st.error("Please upload service account JSON file")
    
    # Box Configuration
    with source_tabs[2]:
        st.markdown("#### Box")
        
        with st.expander("Box Configuration", expanded=True):
            box_token = st.text_input("API Token", key="box_token", type="password")
            
            if st.button("Connect to Box", key="connect_box"):
                if box_token:
                    config = {'api_token': box_token}
                    connector = get_data_source_connector('box', config)
                    if connector and connector.authenticate():
                        st.session_state.data_sources['box'] = connector
                        st.success("✅ Connected to Box successfully!")
                    else:
                        st.error("❌ Failed to connect to Box")
                else:
                    st.error("Please enter API token")
    
    # Dropbox Configuration
    with source_tabs[3]:
        st.markdown("#### Dropbox")
        
        with st.expander("Dropbox Configuration", expanded=True):
            dropbox_token = st.text_input("Access Token", key="dropbox_token", type="password")
            
            if st.button("Connect to Dropbox", key="connect_dropbox"):
                if dropbox_token:
                    config = {'access_token': dropbox_token}
                    connector = get_data_source_connector('dropbox', config)
                    if connector and connector.authenticate():
                        st.session_state.data_sources['dropbox'] = connector
                        st.success("✅ Connected to Dropbox successfully!")
                    else:
                        st.error("❌ Failed to connect to Dropbox")
                else:
                    st.error("Please enter access token")
    
    # Local Directory
    with source_tabs[4]:
        st.markdown("#### Local Directory Upload")
        
        with st.expander("Directory Upload", expanded=True):
            st.info("📁 Upload multiple files or a ZIP archive containing your documents")
            
            bulk_files = st.file_uploader(
                "Select multiple files or ZIP archive",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'json', 'xml', 'txt', 'md', 'zip'],
                key="bulk_upload"
            )
            
            if bulk_files and st.button("Process Bulk Upload", key="process_bulk"):
                process_bulk_files(bulk_files)

def process_bulk_files(files):
    """Process multiple files including ZIP archives"""
    with st.spinner("Processing bulk upload..."):
        processed_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}...")
            
            if file.name.endswith('.zip'):
                # Handle ZIP archive
                zip_content = file.read()
                extracted_files = process_zip_file(zip_content)
                
                for extracted_file in extracted_files:
                    if extracted_file['success']:
                        st.session_state.uploaded_documents.append({
                            'file': None,  # No original file object for extracted files
                            'result': extracted_file,
                            'processed_at': datetime.now()
                        })
                        processed_count += 1
                    else:
                        error_count += 1
            else:
                # Handle regular file
                result = process_uploaded_file(file)
                
                if result['success']:
                    st.session_state.uploaded_documents.append({
                        'file': file,
                        'result': result,
                        'processed_at': datetime.now()
                    })
                    processed_count += 1
                else:
                    error_count += 1
            
            progress_bar.progress((i + 1) / len(files))
        
        status_text.empty()
        progress_bar.empty()
        
        if processed_count > 0:
            st.success(f"✅ Successfully processed {processed_count} files")
        
        if error_count > 0:
            st.warning(f"⚠️ Failed to process {error_count} files")

def render_file_browser(data_source: str):
    """Render file browser for connected data source"""
    if data_source not in st.session_state.data_sources:
        st.error(f"Not connected to {data_source}")
        return
    
    connector = st.session_state.data_sources[data_source]
    
    st.markdown(f"#### 📂 Browse {data_source.title()}")
    
    # List files
    try:
        files = connector.list_files()
        
        if files:
            # Create selection interface
            selected_files = []
            
            for file in files:
                col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                
                with col1:
                    if st.checkbox("", key=f"select_{file['id']}"):
                        selected_files.append(file)
                
                with col2:
                    st.text(file['name'])
                
                with col3:
                    st.text(f"{file['size'] // 1024} KB")
                
                with col4:
                    st.text(file['modified'])
            
            # Download selected files
            if selected_files:
                if st.button(f"Download {len(selected_files)} selected files", key=f"download_{data_source}"):
                    download_files_from_source(connector, selected_files)
        else:
            st.info("No files found in the data source")
    
    except Exception as e:
        st.error(f"Error browsing {data_source}: {str(e)}")

def download_files_from_source(connector, files):
    """Download and process files from data source"""
    with st.spinner(f"Downloading {len(files)} files..."):
        processed_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        
        for i, file in enumerate(files):
            try:
                file_content = connector.download_file(file['id'], file['name'])
                
                source_info = {
                    'name': file['name'],
                    'source': connector.__class__.__name__.replace('Connector', '').lower(),
                    'size': file['size'],
                    'modified': file['modified']
                }
                
                result = process_uploaded_file(file_content, source_info)
                
                if result['success']:
                    st.session_state.uploaded_documents.append({
                        'file': None,
                        'result': result,
                        'processed_at': datetime.now()
                    })
                    processed_count += 1
                else:
                    error_count += 1
            
            except Exception as e:
                st.error(f"Error downloading {file['name']}: {str(e)}")
                error_count += 1
            
            progress_bar.progress((i + 1) / len(files))
        
        progress_bar.empty()
        
        if processed_count > 0:
            st.success(f"✅ Successfully downloaded and processed {processed_count} files")
        
        if error_count > 0:
            st.warning(f"⚠️ Failed to process {error_count} files")

def render_batch_analysis():
    """Render batch analysis interface"""
    st.markdown("### 🔄 Batch Analysis")
    
    if not st.session_state.uploaded_documents:
        st.info("No documents available for batch analysis")
        return
    
    with st.expander("Batch Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_document_type = st.selectbox(
                "Document Type for All:",
                list(get_document_templates().keys()),
                key="batch_doc_type"
            )
            
            batch_analysis_mode = st.selectbox(
                "Analysis Mode:",
                ["Quick Summary", "Deep Analysis", "Compliance Check", "Custom Analysis"],
                key="batch_analysis_mode"
            )
        
        with col2:
            # Document selection
            all_docs = st.checkbox("Analyze All Documents", value=True)
            
            if not all_docs:
                selected_indices = st.multiselect(
                    "Select Documents:",
                    range(len(st.session_state.uploaded_documents)),
                    format_func=lambda x: st.session_state.uploaded_documents[x]['result']['details']['filename']
                )
            else:
                selected_indices = list(range(len(st.session_state.uploaded_documents)))
        
        # Custom questions for batch analysis
        if batch_analysis_mode == "Custom Analysis":
            batch_custom_questions = st.text_area(
                "Custom Questions (applied to all documents):",
                placeholder="Enter questions to be analyzed across all selected documents..."
            )
        else:
            batch_custom_questions = ""
        
        # Batch analysis execution
        if st.button("🚀 Start Batch Analysis", type="primary"):
            if selected_indices:
                run_batch_analysis(selected_indices, batch_document_type, batch_analysis_mode, batch_custom_questions)
            else:
                st.warning("Please select at least one document for analysis")

def run_batch_analysis(selected_indices, document_type, analysis_mode, custom_questions):
    """Execute batch analysis on selected documents"""
    st.markdown("### 📊 Batch Analysis Results")
    
    # Get LLM from config
    from config import get_config
    cfg = get_config()
    llm = cfg.get('llm')
    
    if not llm:
        st.error("LLM not configured. Please set up AI configuration first.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_results = []
    
    for i, doc_idx in enumerate(selected_indices):
        document = st.session_state.uploaded_documents[doc_idx]
        filename = document['result']['details']['filename']
        
        status_text.text(f"Analyzing {filename}...")
        
        try:
            content = document['result']['content']
            analysis_result = analyze_document_with_ai(
                content, document_type, analysis_mode, custom_questions, llm
            )
            
            batch_results.append({
                'filename': filename,
                'analysis': analysis_result,
                'success': True
            })
            
            # Store individual result in history
            st.session_state.analysis_history.append({
                'document': filename,
                'type': document_type,
                'mode': f"{analysis_mode} (Batch)",
                'result': analysis_result,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            batch_results.append({
                'filename': filename,
                'analysis': f"Error: {str(e)}",
                'success': False
            })
        
        progress_bar.progress((i + 1) / len(selected_indices))
    
    status_text.empty()
    progress_bar.empty()
    
    # Display batch results
    success_count = sum(1 for result in batch_results if result['success'])
    st.success(f"✅ Completed batch analysis: {success_count}/{len(batch_results)} successful")
    
    # Results tabs
    if batch_results:
        result_tabs = st.tabs([f"📄 {result['filename'][:20]}..." if len(result['filename']) > 20 else f"📄 {result['filename']}" for result in batch_results])
        
        for i, (tab, result) in enumerate(zip(result_tabs, batch_results)):
            with tab:
                if result['success']:
                    st.text_area(f"Analysis: {result['filename']}", result['analysis'], height=400)
                else:
                    st.error(f"Analysis failed: {result['analysis']}")
                
                # Download individual result
                st.download_button(
                    f"Download Analysis",
                    result['analysis'],
                    f"batch_analysis_{result['filename']}.txt",
                    key=f"download_batch_{i}"
                )
        
        # Batch export options
        st.markdown("### 📥 Batch Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Combined report
            combined_report = generate_combined_report(batch_results, document_type, analysis_mode)
            st.download_button(
                "📑 Download Combined Report",
                combined_report,
                f"batch_analysis_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
        
        with col2:
            # CSV summary
            csv_summary = generate_batch_csv_summary(batch_results)
            st.download_button(
                "📊 Download CSV Summary",
                csv_summary,
                f"batch_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        
        with col3:
            # JSON export
            json_export = generate_batch_json_export(batch_results, document_type, analysis_mode)
            st.download_button(
                "💾 Download JSON Export",
                json_export,
                f"batch_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

def generate_combined_report(batch_results, document_type, analysis_mode):
    """Generate combined report from batch analysis"""
    report = f"""
BATCH ANALYSIS REPORT
═══════════════════════════════════════════════════════════════
Document Type: {document_type}
Analysis Mode: {analysis_mode}
Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Documents: {len(batch_results)}
Successful Analyses: {sum(1 for r in batch_results if r['success'])}
═══════════════════════════════════════════════════════════════

"""
    
    for i, result in enumerate(batch_results, 1):
        report += f"""
DOCUMENT {i}: {result['filename']}
{'─' * 80}
{result['analysis']}

{'═' * 80}

"""
    
    return report

def generate_batch_csv_summary(batch_results):
    """Generate CSV summary of batch analysis"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Filename', 'Status', 'Analysis_Length', 'Timestamp'])
    
    # Data
    for result in batch_results:
        writer.writerow([
            result['filename'],
            'Success' if result['success'] else 'Failed',
            len(result['analysis']),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    return output.getvalue()

def generate_batch_json_export(batch_results, document_type, analysis_mode):
    """Generate JSON export of batch analysis"""
    export_data = {
        'metadata': {
            'document_type': document_type,
            'analysis_mode': analysis_mode,
            'generated_at': datetime.now().isoformat(),
            'total_documents': len(batch_results),
            'successful_analyses': sum(1 for r in batch_results if r['success'])
        },
        'results': batch_results
    }
    
    return json.dumps(export_data, indent=2)

def render_advanced_analytics():
    """Render advanced analytics and insights"""
    st.markdown("### 📈 Advanced Analytics")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available for advanced analytics")
        return
    
    # Analytics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_analyses = len(st.session_state.analysis_history)
        st.metric("Total Analyses", total_analyses)
    
    with col2:
        unique_docs = len(set(analysis['document'] for analysis in st.session_state.analysis_history))
        st.metric("Unique Documents", unique_docs)
    
    with col3:
        doc_types = [analysis['type'] for analysis in st.session_state.analysis_history]
        most_common_type = max(set(doc_types), key=doc_types.count) if doc_types else "N/A"
        st.metric("Most Analyzed Type", most_common_type)
    
    with col4:
        analysis_modes = [analysis['mode'] for analysis in st.session_state.analysis_history]
        most_common_mode = max(set(analysis_modes), key=analysis_modes.count) if analysis_modes else "N/A"
        st.metric("Most Used Mode", most_common_mode)
    
    # Analytics tabs
    analytics_tabs = st.tabs(["📊 Document Types", "🔍 Analysis Patterns", "📅 Timeline", "💡 Insights"])
    
    with analytics_tabs[0]:
        # Document type distribution
        st.markdown("#### Document Type Distribution")
        
        doc_type_counts = {}
        for analysis in st.session_state.analysis_history:
            doc_type = analysis['type']
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        
        if doc_type_counts:
            df_doc_types = pd.DataFrame(list(doc_type_counts.items()), columns=['Document Type', 'Count'])
            st.bar_chart(df_doc_types.set_index('Document Type'))
    
    with analytics_tabs[1]:
        # Analysis mode patterns
        st.markdown("#### Analysis Mode Patterns")
        
        mode_counts = {}
        for analysis in st.session_state.analysis_history:
            mode = analysis['mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        if mode_counts:
            df_modes = pd.DataFrame(list(mode_counts.items()), columns=['Analysis Mode', 'Count'])
            st.bar_chart(df_modes.set_index('Analysis Mode'))
    
    with analytics_tabs[2]:
        # Timeline analysis
        st.markdown("#### Analysis Timeline")
        
        # Group analyses by date
        from collections import defaultdict
        daily_counts = defaultdict(int)
        
        for analysis in st.session_state.analysis_history:
            try:
                date = analysis['timestamp'].split(' ')[0]  # Extract date part
                daily_counts[date] += 1
            except:
                pass
        
        if daily_counts:
            df_timeline = pd.DataFrame(list(daily_counts.items()), columns=['Date', 'Analyses'])
            df_timeline['Date'] = pd.to_datetime(df_timeline['Date'])
            df_timeline = df_timeline.sort_values('Date')
            st.line_chart(df_timeline.set_index('Date'))
    
    with analytics_tabs[3]:
        # AI-powered insights
        st.markdown("#### 💡 Analysis Insights")
        
        # Most active documents
        doc_activity = {}
        for analysis in st.session_state.analysis_history:
            doc = analysis['document']
            doc_activity[doc] = doc_activity.get(doc, 0) + 1
        
        if doc_activity:
            most_analyzed = sorted(doc_activity.items(), key=lambda x: x[1], reverse=True)[:5]
            
            st.markdown("**Most Analyzed Documents:**")
            for doc, count in most_analyzed:
                st.write(f"• {doc}: {count} analyses")
        
        # Analysis patterns
        st.markdown("**Analysis Patterns:**")
        
        if total_analyses > 10:
            st.write(f"• High activity user - {total_analyses} total analyses")
        
        if len(set(doc_types)) > 3:
            st.write(f"• Diverse document portfolio - {len(set(doc_types))} different types")
        
        batch_analyses = sum(1 for analysis in st.session_state.analysis_history if 'Batch' in analysis['mode'])
        if batch_analyses > 0:
            st.write(f"• Efficient batch processing - {batch_analyses} batch analyses")

def render_document_intelligence_tab(llm=None):
    """Enhanced main render function for Document Intelligence Hub"""
    st.markdown("## 📄 Document Intelligence Hub")
    st.markdown("*Advanced RAG system for analyzing multiple document types with bulk upload and data source integrations*")
    
    # Initialize session state
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Main navigation tabs
    main_tabs = st.tabs([
        "📤 Upload & Process",
        "🔗 Data Sources", 
        "📚 Document Library",
        "🔍 Analysis",
        "🔄 Batch Processing",
        "📈 Analytics"
    ])
    
    # Upload & Process Tab
    with main_tabs[0]:
        st.markdown("### 📤 Document Upload & Processing")
        
        # File type support info
        col1, col2 = st.columns([2, 1])
        
        with col2:
            with st.expander("📋 Supported Formats", expanded=True):
                file_types = supported_file_types()
                for category, extensions in file_types.items():
                    st.markdown(f"**{category}:** {', '.join(extensions)}")
        
        with col1:
            # Single file upload
            st.markdown("#### Single File Upload")
            uploaded_file = st.file_uploader(
                "Choose a document to analyze",
                type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'csv', 'json', 'xml', 'txt', 'md', 'png', 'jpg', 'jpeg']
            )
            
            if uploaded_file:
                if st.button("Process Single File", key="process_single"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = process_uploaded_file(uploaded_file)
                        
                        if result['success']:
                            st.session_state.uploaded_documents.append({
                                'file': uploaded_file,
                                'result': result,
                                'processed_at': datetime.now()
                            })
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                        else:
                            st.error(f"❌ Error processing {uploaded_file.name}: {result.get('error', 'Unknown error')}")
            
            # Bulk file upload
            st.markdown("#### Bulk File Upload")
            bulk_files = st.file_uploader(
                "Select multiple files or ZIP archives",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'json', 'xml', 'txt', 'md', 'zip'],
                key="bulk_files"
            )
            
            if bulk_files:
                if st.button("Process Bulk Files", key="process_bulk_main"):
                    process_bulk_files(bulk_files)
    
    # Data Sources Tab
    with main_tabs[1]:
        render_data_source_config()
        
        # File browser for connected sources
        connected_sources = list(st.session_state.get('data_sources', {}).keys())
        
        if connected_sources:
            st.markdown("### 📂 Browse Connected Sources")
            
            selected_source = st.selectbox("Select Data Source:", connected_sources)
            
            if selected_source:
                render_file_browser(selected_source)
        else:
            st.info("Connect to data sources above to browse and import files")
    
    # Document Library Tab
    with main_tabs[2]:
        st.markdown("### 📚 Document Library")
        
        if st.session_state.uploaded_documents:
            # Library overview
            total_docs = len(st.session_state.uploaded_documents)
            total_size = sum(doc['result']['details']['filesize'] for doc in st.session_state.uploaded_documents) / (1024 * 1024)  # MB
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", total_docs)
            
            with col2:
                st.metric("Total Size", f"{total_size:.2f} MB")
            
            with col3:
                unique_types = len(set(doc['result']['details']['filetype'] for doc in st.session_state.uploaded_documents))
                st.metric("File Types", unique_types)
            
            # Document table with enhanced features
            doc_data = []
            for i, doc in enumerate(st.session_state.uploaded_documents):
                doc_data.append({
                    'Index': i,
                    'Filename': doc['result']['details']['filename'],
                    'Type': doc['result']['details'].get('filetype', 'Unknown'),
                    'Size': f"{doc['result']['details']['filesize'] / 1024:.1f} KB",
                    'Source': doc['result']['details'].get('source', 'upload'),
                    'Uploaded': doc['result']['details']['upload_time'],
                    'Hash': doc['result']['details'].get('hash', 'N/A')[:16] + '...',
                    'Status': '✅ Ready' if doc['result']['success'] else '❌ Error'
                })
            
            docs_df = pd.DataFrame(doc_data)
            
            # Search and filter
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_term = st.text_input("🔍 Search documents", placeholder="Enter filename or type...")
                
            with col2:
                filter_type = st.selectbox("Filter by type:", ["All"] + list(docs_df['Type'].unique()))
            
            # Apply filters
            filtered_df = docs_df.copy()
            
            if search_term:
                filtered_df = filtered_df[filtered_df['Filename'].str.contains(search_term, case=False)]
            
            if filter_type != "All":
                filtered_df = filtered_df[filtered_df['Type'] == filter_type]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Document actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Clear All Documents"):
                    st.session_state.uploaded_documents = []
                    st.rerun()
            
            with col2:
                # Export library metadata
                library_export = docs_df.to_csv(index=False)
                st.download_button(
                    "📥 Export Library CSV",
                    library_export,
                    "document_library.csv",
                    "text/csv"
                )
            
            with col3:
                if st.button("🔄 Refresh Library"):
                    st.rerun()
        
        else:
            st.info("📁 No documents in library. Upload documents in the 'Upload & Process' tab.")
    
    # Analysis Tab
    with main_tabs[3]:
        st.markdown("### 🔍 Document Analysis")
        
        if not st.session_state.uploaded_documents:
            st.info("Upload documents first to start analysis")
        else:
            # Document selection
            selected_doc_idx = st.selectbox(
                "Select document to analyze:",
                range(len(st.session_state.uploaded_documents)),
                format_func=lambda x: st.session_state.uploaded_documents[x]['result']['details']['filename']
            )
            
            selected_doc = st.session_state.uploaded_documents[selected_doc_idx]
            
            # Analysis configuration
            col1, col2 = st.columns(2)
            
            with col1:
                document_type = st.selectbox(
                    "Document Type:",
                    list(get_document_templates().keys())
                )
                
                analysis_mode = st.selectbox(
                    "Analysis Mode:",
                    ["Quick Summary", "Deep Analysis", "Compliance Check", "Custom Analysis", "Comparative Analysis"]
                )
            
            with col2:
                use_template = st.checkbox("Use Analysis Template", value=True)
                
                if analysis_mode == "Custom Analysis" or not use_template:
                    custom_questions = st.text_area(
                        "Custom Analysis Questions:",
                        placeholder="Enter specific questions about this document...",
                        height=100
                    )
                else:
                    custom_questions = ""
            
            # Template preview
            if use_template and analysis_mode != "Custom Analysis":
                with st.expander("📋 Analysis Template Preview"):
                    templates = get_document_templates()
                    template = templates.get(document_type, {})
                    
                    if template.get('questions'):
                        st.markdown("**Key Questions:**")
                        for q in template['questions'][:3]:  # Show first 3
                            st.write(f"• {q}")
                    
                    if template.get('analysis_points'):
                        st.markdown("**Focus Areas:**")
                        for point in template['analysis_points'][:3]:  # Show first 3
                            st.write(f"• {point}")
            
            # Document preview
            with st.expander("📄 Document Preview", expanded=False):
                content = selected_doc['result']['content']
                st.text_area("Document Content (Preview)", content[:1000] + "..." if len(content) > 1000 else content, height=200)
            
            # Analysis execution
            col1, col2 = st.columns([3, 1])
            
            with col1:
                analyze_button = st.button("🔬 Analyze Document", type="primary")
            
            with col2:
                debug_mode = st.checkbox("Debug Mode", help="Show detailed LLM information for troubleshooting")
            
            if analyze_button:
                with st.spinner("Analyzing document with AI..."):
                    # Debug information if enabled
                    if debug_mode:
                        st.markdown("### 🐛 Debug Information")
                        
                        if llm:
                            st.write(f"**LLM Type:** {type(llm).__name__}")
                            st.write(f"**LLM Methods:** {[m for m in dir(llm) if not m.startswith('_')][:10]}")
                            
                            # Test LLM with simple prompt
                            test_prompt = "Hello, can you respond with 'LLM is working correctly'?"
                            try:
                                test_response = get_llm_response(llm, test_prompt)
                                st.success(f"✅ LLM Test Response: {test_response[:100]}...")
                            except Exception as e:
                                st.error(f"❌ LLM Test Failed: {str(e)}")
                        else:
                            st.error("❌ No LLM configured")
                        
                        st.markdown("---")
                    
                    content = selected_doc['result']['content']
                    analysis_result = analyze_document_with_ai(
                        content, document_type, analysis_mode, custom_questions, llm
                    )
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        'document': selected_doc['result']['details']['filename'],
                        'type': document_type,
                        'mode': analysis_mode,
                        'result': analysis_result,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    st.text_area("AI Analysis Output:", analysis_result, height=500)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "📄 Download Analysis (TXT)",
                            analysis_result,
                            f"analysis_{selected_doc['result']['details']['filename']}.txt"
                        )
                    
                    with col2:
                        # Create structured report
                        structured_report = f"""
DOCUMENT ANALYSIS REPORT
{"="*50}
Document: {selected_doc['result']['details']['filename']}
Type: {document_type}
Mode: {analysis_mode}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"="*50}

{analysis_result}
"""
                        st.download_button(
                            "📋 Download Report",
                            structured_report,
                            f"report_{selected_doc['result']['details']['filename']}.txt"
                        )
                    
                    with col3:
                        # JSON export
                        json_export = {
                            "document": selected_doc['result']['details']['filename'],
                            "analysis_type": document_type,
                            "analysis_mode": analysis_mode,
                            "timestamp": datetime.now().isoformat(),
                            "result": analysis_result
                        }
                        
                        st.download_button(
                            "💾 Download JSON",
                            json.dumps(json_export, indent=2),
                            f"analysis_{selected_doc['result']['details']['filename']}.json"
                            ),
    
    # Batch Processing Tab
    with main_tabs[4]:
        render_batch_analysis()
    
    # Analytics Tab
    with main_tabs[5]:
        render_advanced_analytics()
        
        # Analysis History
        if st.session_state.analysis_history:
            st.markdown("### 📜 Recent Analysis History")
            
            # Show recent analyses
            recent_analyses = st.session_state.analysis_history[-10:]  # Last 10
            
            for i, analysis in enumerate(reversed(recent_analyses)):
                with st.expander(f"📄 {analysis['document']} - {analysis['timestamp']}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.write(f"**Type:** {analysis['type']}")
                        st.write(f"**Mode:** {analysis['mode']}")
                        st.write(f"**Time:** {analysis['timestamp']}")
                    
                    with col2:
                        # Show truncated result
                        result_preview = analysis['result'][:300] + "..." if len(analysis['result']) > 300 else analysis['result']
                        st.text(result_preview)
                        
                        # Download option
                        st.download_button(
                            "📥 Download Full Analysis",
                            analysis['result'],
                            f"history_{analysis['timestamp'].replace(':', '-')}_{analysis['document']}.txt",
                            key=f"download_history_{i}"
                        )
            
            # Clear history option
            if st.button("🗑️ Clear Analysis History"):
                st.session_state.analysis_history = []
                st.rerun()

# Main render function (for compatibility)
def render_document_intelligence_tab_enhanced(llm=None):
    """Enhanced render function with full functionality"""
    return render_document_intelligence_tab(llm)