#!/usr/bin/env python3
"""
Quick fix script to resolve circular import issues.
This script will check for and fix common circular import problems.
"""

import os
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file before modifying."""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"📁 Backed up {filepath} to {backup_path}")
        return True
    return False

def check_circular_import_in_file(filepath):
    """Check if a file has potential circular import issues."""
    if not os.path.exists(filepath):
        return False, []
    
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            # Check for self-imports
            if 'from services.data_fetcher import' in line and 'services/data_fetcher.py' in filepath:
                issues.append(f"Line {i}: Self-import detected - {line}")
            
            # Check for problematic import patterns
            if 'from ui.tabs' in line and 'ui/' in filepath:
                issues.append(f"Line {i}: Potential circular import - {line}")
                
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return len(issues) > 0, issues

def fix_data_fetcher():
    """Fix the data_fetcher.py file to remove circular imports."""
    filepath = "services/data_fetcher.py"
    
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found")
        return False
    
    # Backup first
    backup_file(filepath)
    
    # Read current content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it has the problematic self-import
        if 'from services.data_fetcher import' in content:
            print(f"🔧 Fixing circular import in {filepath}")
            
            # Remove the problematic line
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if 'from services.data_fetcher import' not in line:
                    fixed_lines.append(line)
                else:
                    print(f"   ❌ Removed: {line.strip()}")
            
            # Write fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            
            print(f"✅ Fixed {filepath}")
            return True
        else:
            print(f"✅ {filepath} looks clean")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def scan_for_circular_imports():
    """Scan all Python files for potential circular import issues."""
    print("🔍 Scanning for circular import issues...")
    
    files_to_check = [
        "main.py",
        "config.py",
        "services/data_fetcher.py",
        "ui/sidebar.py",
        "ui/tabs/market_analysis.py",
        "ui/tabs/portfolio.py",
        "ui/tabs/pro_dashboard.py",
        "ui/tabs/auto_trading.py",
        "analysis/technical.py",
        "analysis/predictive.py",
        "analysis/reporting.py",
        "core/trading_engine.py",
        "core/trading_bot.py"
    ]
    
    issues_found = False
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            has_issues, issues = check_circular_import_in_file(filepath)
            if has_issues:
                print(f"⚠️  Issues in {filepath}:")
                for issue in issues:
                    print(f"   - {issue}")
                issues_found = True
            else:
                print(f"✅ {filepath}")
        else:
            print(f"⚠️  {filepath} not found")
    
    return not issues_found

def test_imports():
    """Test critical imports to see if they work."""
    print("\n🧪 Testing imports...")
    
    test_cases = [
        ("config", "get_config"),
        ("services.data_fetcher", "get_market_data_yfinance"),
        ("services.data_fetcher", "fetch_all_tickers"),
        ("ui.sidebar", "render_sidebar"),
        ("core.trading_engine", "AutoTradingEngine"),
    ]
    
    all_passed = True
    
    for module_name, function_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)
            print(f"✅ {module_name}.{function_name}")
        except Exception as e:
            print(f"❌ {module_name}.{function_name}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main function to run all fixes."""
    print("🚀 AI Trading Platform - Circular Import Fix")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Step 1: Scan for issues
    print("\n📋 Step 1: Scanning for circular import issues")
    clean_scan = scan_for_circular_imports()
    
    # Step 2: Fix data_fetcher if needed
    print("\n🔧 Step 2: Fixing data_fetcher.py")
    data_fetcher_fixed = fix_data_fetcher()
    
    # Step 3: Test imports
    print("\n🧪 Step 3: Testing imports")
    imports_working = test_imports()
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Scan clean: {'✅' if clean_scan else '❌'}")
    print(f"   Data fetcher fixed: {'✅' if data_fetcher_fixed else '❌'}")
    print(f"   Imports working: {'✅' if imports_working else '❌'}")
    
    if imports_working:
        print("\n🎉 All fixes applied successfully!")
        print("\nYou can now run:")
        print("   streamlit run main.py")
    else:
        print("\n⚠️  Some issues remain. Check the error messages above.")
        print("\nYou may need to:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check for typos in import statements")
        print("   3. Verify file paths and module structure")
    
    return imports_working

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)