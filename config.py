import os
from typing import Optional, Dict, Any
import streamlit as st

class Config:
    """
    Configuration management class for API keys and settings.
    Supports environment variables, config file, and fallback defaults.
    """
    
    def __init__(self):
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and defaults"""
        
        # Load .env file if it exists
        self._load_env_file()
        
        # EODHD API Configuration
        self._config['eodhd'] = {
            'api_key': os.getenv('EODHD_API_KEY', ''),
            'base_url': os.getenv('EODHD_BASE_URL', 'https://eodhd.com/api'),
            'timeout': int(os.getenv('EODHD_TIMEOUT', '30'))
        }
        
        # Azure OpenAI Configuration
        self._config['azure_openai'] = {
            'api_key': os.getenv('AZURE_OPENAI_API_KEY', ''),
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT', ''),
            'chat_deployment': os.getenv('AZURE_CHAT_DEPLOYMENT', ''),
            'whisper_deployment': os.getenv('AZURE_WHISPER_DEPLOYMENT', ''),
            'api_version': os.getenv('AZURE_API_VERSION', '2024-02-01'),
            'temperature': float(os.getenv('AZURE_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('AZURE_MAX_TOKENS', '4000'))
        }
        
        # Trading Configuration
        self._config['trading'] = {
            'default_symbols': ['AAPL.US', 'GOOGL.US', 'MSFT.US', 'TSLA.US', 'AMZN.US'],
            'max_portfolio_value': float(os.getenv('MAX_PORTFOLIO_VALUE', '1000000')),
            'risk_tolerance': float(os.getenv('RISK_TOLERANCE', '0.02')),
            'enable_paper_trading': os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
        }
        
        # Application Settings
        self._config['app'] = {
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'use_cache': os.getenv('USE_CACHE', 'true').lower() == 'true',
            'cache_timeout': int(os.getenv('CACHE_TIMEOUT', '300')),
            'max_tickers_display': int(os.getenv('MAX_TICKERS_DISPLAY', '100'))
        }
        
        # Authentication Settings
        self._config['auth'] = {
            'demo_username': os.getenv('DEMO_USERNAME', 'genaiwithprabhakar'),
            'demo_password': os.getenv('DEMO_PASSWORD', 'genaiwithprabhakar'),
            'enable_guest_mode': os.getenv('ENABLE_GUEST_MODE', 'true').lower() == 'true'
        }
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists"""
        try:
            from dotenv import load_dotenv
            # Try to load .env file
            if os.path.exists('.env'):
                load_dotenv('.env')
                return True
            else:
                # Also try parent directory
                if os.path.exists('../.env'):
                    load_dotenv('../.env')
                    return True
        except ImportError:
            # python-dotenv not installed, that's okay
            pass
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
        return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            # First check session state overrides
            try:
                if hasattr(st, 'session_state') and 'config_overrides' in st.session_state:
                    overrides = st.session_state.config_overrides
                    if section in overrides and key in overrides[section]:
                        return overrides[section][key]
            except:
                # If streamlit is not available or session_state doesn't exist, continue
                pass
            
            return self._config.get(section, {}).get(key, default)
        except:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self._config.get(section, {})
    
    def is_configured(self, section: str, required_keys: list = None) -> bool:
        """Check if a section is properly configured"""
        section_config = self.get_section(section)
        
        if not section_config:
            return False
        
        if required_keys:
            return all(section_config.get(key) for key in required_keys)
        
        # Default required keys for common sections
        default_required = {
            'eodhd': ['api_key'],
            'azure_openai': ['api_key', 'endpoint', 'chat_deployment']
        }
        
        required = default_required.get(section, [])
        return all(section_config.get(key) for key in required)
    
    def get_eodhd_api_key(self) -> str:
        """Get EODHD API key"""
        return self.get('eodhd', 'api_key', '')
    
    def get_azure_config(self) -> Dict[str, str]:
        """Get complete Azure OpenAI configuration"""
        return self.get_section('azure_openai')
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.get_section('trading')
    
    def get_auth_config(self) -> Dict[str, str]:
        """Get authentication configuration"""
        return self.get_section('auth')
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate all configuration sections"""
        validation_results = {}
        
        # Validate EODHD
        validation_results['eodhd'] = self.is_configured('eodhd', ['api_key'])
        
        # Validate Azure OpenAI
        validation_results['azure_openai'] = self.is_configured(
            'azure_openai', 
            ['api_key', 'endpoint', 'chat_deployment']
        )
        
        # App config is always valid (has defaults)
        validation_results['app'] = True
        validation_results['trading'] = True
        validation_results['auth'] = True
        
        return validation_results
    
    def get_status_summary(self) -> str:
        """Get a summary of configuration status"""
        validation = self.validate_config()
        
        status_parts = []
        if validation['eodhd']:
            status_parts.append("✅ EODHD")
        else:
            status_parts.append("❌ EODHD")
        
        if validation['azure_openai']:
            status_parts.append("✅ Azure AI")
        else:
            status_parts.append("❌ Azure AI")
        
        return " | ".join(status_parts)
    
    def debug_config(self):
        """Debug function to show current configuration status"""
        print("🔍 Configuration Debug Information")
        print("=" * 50)
        
        # Check .env file
        env_exists = os.path.exists('.env')
        print(f".env file exists: {env_exists}")
        
        if env_exists:
            try:
                with open('.env', 'r') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    env_vars = [line for line in lines if '=' in line and not line.strip().startswith('#')]
                    print(f".env file has {len(env_vars)} variables")
            except Exception as e:
                print(f"Error reading .env: {e}")
        
        # Check environment variables
        print("\nEnvironment Variables:")
        for section_name, section_data in self._config.items():
            print(f"\n[{section_name}]")
            for key, value in section_data.items():
                # Mask sensitive data
                if 'key' in key.lower() or 'password' in key.lower():
                    if value:
                        masked_value = value[:8] + "..." if len(str(value)) > 8 else "***"
                        print(f"  {key}: {masked_value}")
                    else:
                        print(f"  {key}: NOT SET")
                else:
                    print(f"  {key}: {value}")
        
        # Validation results
        print(f"\nValidation: {self.get_status_summary()}")


# Global configuration instance
config = Config()


# =================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =================================================================

def get_config():
    """Backward compatibility function - returns the global config instance"""
    return config

def get_option(section, key, default=None):
    """Backward compatibility function for getting configuration options"""
    return config.get(section, key, default)

def set_option(section, key, value):
    """Backward compatibility function for setting configuration options (session only)"""
    try:
        if 'config_overrides' not in st.session_state:
            st.session_state.config_overrides = {}
        if section not in st.session_state.config_overrides:
            st.session_state.config_overrides[section] = {}
        st.session_state.config_overrides[section][key] = value
    except:
        pass

def get_config_option(key, default=None):
    """Get config option using dot notation (section.key)"""
    try:
        if '.' in key:
            section, option = key.split('.', 1)
            return config.get(section, option, default)
        else:
            # If no section specified, try to find in any section
            for section_name, section_data in config._config.items():
                if key in section_data:
                    return section_data[key]
            return default
    except Exception:
        return default

def set_config_option(key, value):
    """Set config option in session state (temporary override)"""
    try:
        if 'config_overrides' not in st.session_state:
            st.session_state.config_overrides = {}
            
        if '.' in key:
            section, option = key.split('.', 1)
            if section not in st.session_state.config_overrides:
                st.session_state.config_overrides[section] = {}
            st.session_state.config_overrides[section][option] = value
        else:
            if 'general' not in st.session_state.config_overrides:
                st.session_state.config_overrides['general'] = {}
            st.session_state.config_overrides['general'][key] = value
    except Exception as e:
        print(f"Warning: Could not set config option {key}: {e}")


# =================================================================
# SETUP AND UTILITY FUNCTIONS
# =================================================================

def load_env_file():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        if os.path.exists('.env'):
            load_dotenv()
            return True
        return False
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
        return False

def create_sample_env():
    """Create a sample .env file if it doesn't exist"""
    if os.path.exists('.env'):
        return False
    
    sample_content = """# Agent RICH Configuration
# Replace with your actual API keys

# EODHD Configuration
EODHD_API_KEY=your_eodhd_api_key_here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_WHISPER_DEPLOYMENT=your_whisper_deployment_name
AZURE_API_VERSION=2024-02-01
AZURE_TEMPERATURE=0.7
AZURE_MAX_TOKENS=4000

# Trading Configuration
MAX_PORTFOLIO_VALUE=1000000
RISK_TOLERANCE=0.02
ENABLE_PAPER_TRADING=true

# Application Settings
DEBUG_MODE=false
USE_CACHE=true
CACHE_TIMEOUT=300
MAX_TICKERS_DISPLAY=100

# Authentication Settings
DEMO_USERNAME=genaiwithprabhakar
DEMO_PASSWORD=genaiwithprabhakar
ENABLE_GUEST_MODE=true
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(sample_content)
        print("✅ Created sample .env file. Please edit it with your actual API keys.")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

# Initialize configuration on import
if __name__ != "__main__":
    # Load .env file when module is imported
    load_env_file()

# Debug function for testing
def test_config():
    """Test configuration and display debug info"""
    print("🧪 Testing Agent RICH Configuration")
    print("=" * 40)
    
    config.debug_config()
    
    print("\n🔍 Quick Tests:")
    try:
        # Test basic access
        debug_mode = config.get('app', 'debug_mode', False)
        print(f"✅ Basic config access: debug_mode = {debug_mode}")
        
        # Test compatibility
        cfg = get_config()
        print(f"✅ get_config() works: {type(cfg)}")
        
        # Test validation
        validation = config.validate_config()
        print(f"✅ Validation works: {validation}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_config()