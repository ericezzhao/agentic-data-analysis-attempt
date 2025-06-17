"""
Test Data Analysis Agent

This module tests the intelligent data analysis agent functionality.
"""

import asyncio
import tempfile
import pandas as pd
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.data_analysis_agent import create_data_analysis_agent
from config.settings import get_settings

logger = logging.getLogger(__name__)


async def test_agent_basic_functionality():
    """Test basic agent functionality."""
    print("🧪 Testing Data Analysis Agent - Basic Functionality")
    print("-" * 50)
    
    try:
        # Create agent
        agent = create_data_analysis_agent(
            session_id="test_session_basic",
            debug_mode=True
        )
        
        print(f"✅ Agent created successfully")
        print(f"   Session ID: {agent.session_id}")
        print(f"   ChromaDB client: {'✓' if agent.chroma_client else '✗'}")
        print(f"   Data processor: {'✓' if agent.data_processor else '✗'}")
        print(f"   Query optimizer: {'✓' if agent.query_optimizer else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent basic functionality test failed: {e}")
        return False


async def test_dataset_processing():
    """Test dataset processing capabilities."""
    print("\n🧪 Testing Data Analysis Agent - Dataset Processing")
    print("-" * 50)
    
    try:
        # Create sample data
        data = {
            'id': range(1, 21),
            'name': [f'Item_{i}' for i in range(1, 21)],
            'category': ['A', 'B', 'C', 'D'] * 5,
            'value': [10.5 + i * 2.3 for i in range(20)],
            'score': [85 + i % 10 for i in range(20)]
        }
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        # Create agent
        agent = create_data_analysis_agent(session_id="test_session_processing")
        
        # Test dataset analysis
        result = await agent.analyze_dataset(
            file_path=temp_file,
            file_name="test_data.csv",
            description="Sample test dataset"
        )
        
        if result["success"]:
            print("✅ Dataset processing successful")
            print(f"   Chunks created: {result.get('chunks_created', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time', 'N/A'):.2f}s")
            print(f"   Insights generated: {len(result.get('insights', []))}")
            
            # Display insights
            for i, insight in enumerate(result.get('insights', [])[:3], 1):
                print(f"   Insight {i}: {insight}")
            
            return True
        else:
            print(f"❌ Dataset processing failed: {result.get('message', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Dataset processing test failed: {e}")
        return False


async def test_query_capabilities():
    """Test query processing capabilities."""
    print("\n🧪 Testing Data Analysis Agent - Query Processing")
    print("-" * 50)
    
    try:
        # Create sample data first
        data = {
            'product_id': range(1, 51),
            'product_name': [f'Product_{i}' for i in range(1, 51)],
            'category': ['Electronics', 'Clothing', 'Books'] * 17,
            'price': [20.0 + i * 1.5 for i in range(50)],
            'rating': [3.0 + (i % 5) * 0.4 for i in range(50)]
        }
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        # Create agent and upload dataset
        agent = create_data_analysis_agent(session_id="test_session_query")
        
        upload_result = await agent.analyze_dataset(
            file_path=temp_file,
            file_name="query_test.csv",
            description="Dataset for query testing"
        )
        
        if not upload_result["success"]:
            print(f"❌ Dataset upload failed: {upload_result.get('message')}")
            return False
        
        print("✅ Dataset uploaded successfully")
        
        # Test queries
        test_queries = [
            "What products are in the Electronics category?",
            "Show me high-rated products",
            "What is the average price?"
        ]
        
        successful_queries = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            result = await agent.query_data(
                query=query,
                dataset_name="query_test.csv"
            )
            
            if result["success"]:
                successful_queries += 1
                print(f"   ✅ Found {result.get('results_count', 0)} results")
                
                # Display insights
                insights = result.get('insights', [])
                if insights:
                    print(f"   💡 {insights[0]}")
            else:
                print(f"   ❌ Query failed: {result.get('message', 'Unknown error')}")
        
        print(f"\n✅ Query processing test completed: {successful_queries}/{len(test_queries)} queries successful")
        return successful_queries > 0
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        return False


async def test_conversation_memory():
    """Test conversation memory capabilities."""
    print("\n🧪 Testing Data Analysis Agent - Conversation Memory")
    print("-" * 50)
    
    try:
        # Create agent
        agent = create_data_analysis_agent(session_id="test_session_memory")
        
        # Get conversation summary
        summary = await agent.get_conversation_summary()
        
        print("✅ Conversation memory test successful")
        print(f"   Session ID: {summary.get('session_id', 'N/A')}")
        print(f"   Total messages: {summary.get('total_messages', 0)}")
        print(f"   User queries: {summary.get('user_queries', 0)}")
        print(f"   Datasets analyzed: {len(summary.get('datasets_analyzed', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation memory test failed: {e}")
        return False


async def run_comprehensive_agent_test():
    """Run comprehensive agent testing."""
    print("🚀 Starting Comprehensive Data Analysis Agent Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_agent_basic_functionality),
        ("Dataset Processing", test_dataset_processing),
        ("Query Processing", test_query_capabilities),
        ("Conversation Memory", test_conversation_memory)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Data Analysis Agent is fully functional.")
        return True
    elif passed_tests > 0:
        print("⚠️  Some tests passed. Agent has partial functionality.")
        return True
    else:
        print("❌ All tests failed. Agent needs debugging.")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive test
    asyncio.run(run_comprehensive_agent_test()) 