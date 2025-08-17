# ui/tabs/document_intelligence.py
"""
Document Intelligence Hub - Advanced RAG system for multiple document types
"""
import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime
from typing import Dict, List, Any

def supported_file_types():
    """Return supported file types"""
    return {
        'PDF': ['.pdf'],
        'Word': ['.docx', '.doc'],
        'Excel': ['.xlsx', '.xls'],
        'CSV': ['.csv'],
        'JSON': ['.json'],
        'XML': ['.xml'],
        'Text': ['.txt', '.md']
    }

def get_document_templates():
    """Return analysis templates for different document types"""
    return {
        'Financial Statements': {
            'questions': [
                "What is the revenue growth year-over-year?",
                "What are the key profitability metrics?",
                "What is the debt-to-equity ratio?",
                "Are there any significant one-time items?",
                "What are the main risk factors mentioned?"
            ],
            'analysis_points': [
                "Revenue trends and seasonality",
                "Margin analysis and cost structure",
                "Balance sheet strength",
                "Cash flow generation",
                "Working capital management"
            ]
        },
        'Legal Documents': {
            'questions': [
                "What are the key terms and conditions?",
                "Are there any material risks or liabilities?",
                "What are the governing laws and jurisdiction?",
                "What are the termination clauses?",
                "Are there any compliance requirements?"
            ],
            'analysis_points': [
                "Contract terms and obligations",
                "Risk allocation and indemnification",
                "Intellectual property provisions",
                "Confidentiality and data protection",
                "Dispute resolution mechanisms"
            ]
        },
        'Due Diligence Reports': {
            'questions': [
                "What are the main investment highlights?",
                "What are the identified risks and mitigants?",
                "What is the market opportunity and competition?",
                "What is the management team assessment?",
                "What are the key assumptions in projections?"
            ],
            'analysis_points': [
                "Market position and competitive advantages",
                "Financial performance and projections",
                "Operational efficiency and scalability",
                "Management quality and governance",
                "ESG factors and sustainability"
            ]
        },
        'Board Materials': {
            'questions': [
                "What are the key decisions required?",
                "What is the financial performance update?",
                "What are the strategic initiatives discussed?",
                "Are there any regulatory or compliance issues?",
                "What are the risk management updates?"
            ],
            'analysis_points': [
                "Strategic direction and initiatives",
                "Financial performance vs budget",
                "Operational metrics and KPIs",
                "Risk management and compliance",
                "Governance and board effectiveness"
            ]
        },
        'ESG Reports': {
            'questions': [
                "What are the key ESG metrics and targets?",
                "How is the company addressing climate risks?",
                "What are the diversity and inclusion initiatives?",
                "What is the stakeholder engagement approach?",
                "Are there any ESG-related controversies?"
            ],
            'analysis_points': [
                "Environmental impact and climate strategy",
                "Social responsibility and community impact",
                "Governance structure and ethics",
                "Stakeholder engagement and materiality",
                "ESG risk management and reporting"
            ]
        }
    }

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract content"""
    try:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Read file content based on type
        content = ""
        structured_data = None
        
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/json":
            json_data = json.load(uploaded_file)
            content = json.dumps(json_data, indent=2)
            structured_data = json_data
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            content = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n\nColumns: {', '.join(df.columns)}\n\nFirst 5 rows:\n{df.head().to_string()}"
            structured_data = df
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            content = f"Excel file with {len(df)} rows and {len(df.columns)} columns.\n\nColumns: {', '.join(df.columns)}\n\nFirst 5 rows:\n{df.head().to_string()}"
            structured_data = df
        else:
            content = f"File type {uploaded_file.type} - Content extraction not implemented yet."
        
        return {
            "details": file_details,
            "content": content,
            "structured_data": structured_data,
            "success": True
        }
    
    except Exception as e:
        return {
            "details": {"filename": uploaded_file.name if uploaded_file else "Unknown"},
            "content": "",
            "structured_data": None,
            "success": False,
            "error": str(e)
        }

def analyze_document_content(content: str, document_type: str, analysis_mode: str, llm=None):
    """Analyze document content using AI"""
    if not llm:
        return "AI analysis not available. Please configure LLM in settings."
    
    templates = get_document_templates()
    template = templates.get(document_type, templates['Financial Statements'])
    
    if analysis_mode == "Quick Summary":
        prompt = f"""
        Provide a concise summary of this {document_type} document:
        
        {content[:2000]}...
        
        Focus on:
        - Key highlights and main points
        - Important metrics or figures
        - Critical decisions or recommendations
        """
    
    elif analysis_mode == "Deep Analysis":
        questions = "\n".join([f"- {q}" for q in template['questions']])
        analysis_points = "\n".join([f"- {point}" for point in template['analysis_points']])
        
        prompt = f"""
        Conduct a comprehensive analysis of this {document_type} document:
        
        {content[:3000]}...
        
        Please address these key questions:
        {questions}
        
        Focus your analysis on:
        {analysis_points}
        
        Provide specific insights, metrics, and actionable recommendations.
        """
    
    elif analysis_mode == "Compliance Check":
        prompt = f"""
        Review this {document_type} for compliance and risk factors:
        
        {content[:2000]}...
        
        Check for:
        - Regulatory compliance issues
        - Risk factors and exposures
        - Missing required disclosures
        - Unusual items or red flags
        
        Provide a compliance assessment with specific recommendations.
        """
    
    try:
        # Mock AI response for now - replace with actual LLM call
        response = f"""
        Analysis Mode: {analysis_mode}
        Document Type: {document_type}
        
        [Mock AI Analysis - Replace with actual LLM integration]
        
        Based on the document content, here are the key findings:
        
        1. **Summary**: This appears to be a {document_type.lower()} with standard structure and content.
        
        2. **Key Metrics**: The document contains numerical data and financial information that requires detailed review.
        
        3. **Risk Assessment**: No immediate red flags identified in the preliminary review.
        
        4. **Recommendations**: 
           - Conduct deeper analysis of specific sections
           - Verify key assumptions and calculations
           - Cross-reference with historical data
        
        Note: This is a placeholder response. Actual AI analysis would provide detailed insights based on the document content.
        """
        
        return response
    
    except Exception as e:
        return f"Error in AI analysis: {str(e)}"

def render_document_intelligence_tab(llm=None):
    """Render the Document Intelligence Hub tab"""
    st.markdown("## 📄 Document Intelligence Hub")
    st.markdown("Advanced RAG system for analyzing multiple document types")
    
    # Initialize session state for documents
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # File Upload Section
    st.markdown("### 📤 Document Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents for analysis",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'json', 'xml', 'txt', 'md']
        )
    
    with col2:
        st.markdown("**Supported Formats:**")
        file_types = supported_file_types()
        for category, extensions in file_types.items():
            st.text(f"{category}: {', '.join(extensions)}")
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in [doc['file'] for doc in st.session_state.uploaded_documents]:
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
    
    # Document Management Section
    if st.session_state.uploaded_documents:
        st.markdown("### 📚 Document Library")
        
        # Display uploaded documents
        doc_data = []
        for i, doc in enumerate(st.session_state.uploaded_documents):
            doc_data.append({
                'Index': i,
                'Filename': doc['result']['details']['filename'],
                'Type': doc['result']['details']['filetype'],
                'Size': f"{doc['result']['details']['filesize'] / 1024:.1f} KB",
                'Uploaded': doc['result']['details']['upload_time'],
                'Status': '✅ Ready' if doc['result']['success'] else '❌ Error'
            })
        
        docs_df = pd.DataFrame(doc_data)
        st.dataframe(docs_df, use_container_width=True)
        
        # Document Analysis Section
        st.markdown("### 🔍 Document Analysis")
        
        # Select document for analysis
        selected_doc_idx = st.selectbox(
            "Select document to analyze:",
            range(len(st.session_state.uploaded_documents)),
            format_func=lambda x: st.session_state.uploaded_documents[x]['result']['details']['filename']
        )
        
        selected_doc = st.session_state.uploaded_documents[selected_doc_idx]
        
        # Analysis configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            document_type = st.selectbox(
                "Document Type:",
                list(get_document_templates().keys())
            )
        
        with col2:
            analysis_mode = st.selectbox(
                "Analysis Mode:",
                ["Quick Summary", "Deep Analysis", "Compliance Check", "Comparative Analysis"]
            )
        
        with col3:
            use_template = st.checkbox("Use Analysis Template", value=True)
        
        # Custom questions
        if not use_template:
            custom_questions = st.text_area(
                "Custom Analysis Questions:",
                placeholder="Enter specific questions you want answered about this document..."
            )
        
        # Analyze button
        if st.button("🔬 Analyze Document", type="primary"):
            with st.spinner("Analyzing document..."):
                content = selected_doc['result']['content']
                analysis_result = analyze_document_content(
                    content, document_type, analysis_mode, llm
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
                st.text_area("Analysis Output:", analysis_result, height=400)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "Download Analysis",
                        analysis_result,
                        f"analysis_{selected_doc['result']['details']['filename']}.txt"
                    )
    
    else:
        st.info("Upload documents to begin analysis")
    
    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("### 📜 Analysis History")
        
        with st.expander("View Previous Analyses", expanded=False):
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
                st.markdown(f"**{analysis['timestamp']}** - {analysis['document']} ({analysis['type']}, {analysis['mode']})")
                with st.expander(f"View Analysis #{len(st.session_state.analysis_history) - i}"):
                    st.text(analysis['result'][:500] + "..." if len(analysis['result']) > 500 else analysis['result'])
    
    # Comparative Analysis Section
    if len(st.session_state.uploaded_documents) > 1:
        st.markdown("### 📊 Comparative Analysis")
        
        compare_docs = st.multiselect(
            "Select documents to compare:",
            range(len(st.session_state.uploaded_documents)),
            format_func=lambda x: st.session_state.uploaded_documents[x]['result']['details']['filename']
        )
        
        if len(compare_docs) >= 2:
            comparison_type = st.selectbox(
                "Comparison Type:",
                ["Financial Performance", "Risk Assessment", "Key Terms", "Custom Comparison"]
            )
            
            if st.button("🔄 Compare Documents"):
                st.info("Comparative analysis feature coming soon...")
    
    # Document Verification Section
    st.markdown("### 🔐 Document Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Authenticity Checks**")
        if st.button("Verify Document Integrity"):
            st.info("Document integrity verification coming soon...")
    
    with col2:
        st.markdown("**Compliance Validation**")
        compliance_framework = st.selectbox(
            "Select Framework:",
            ["SOX Compliance", "GDPR", "ESG Standards", "Investment Covenants"]
        )
        
        if st.button("Run Compliance Check"):
            st.info(f"Compliance check for {compliance_framework} coming soon...")
    
    # Advanced Features
    st.markdown("### ⚙️ Advanced Features")
    
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Batch Processing**")
            if st.button("Analyze All Documents"):
                st.info("Batch analysis feature coming soon...")
            
            st.markdown("**Template Management**")
            if st.button("Create Custom Template"):
                st.info("Custom template creation coming soon...")
        
        with col2:
            st.markdown("**Export Options**")
            export_format = st.selectbox("Export Format:", ["PDF Report", "Excel Summary", "JSON Data"])
            
            if st.button("Export Analysis"):
                st.info(f"Export to {export_format} coming soon...")
    
    # Clear data option
    if st.session_state.uploaded_documents or st.session_state.analysis_history:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear All Data", help="This will remove all uploaded documents and analysis history"):
                st.session_state.uploaded_documents = []
                st.session_state.analysis_history = []
                st.rerun()