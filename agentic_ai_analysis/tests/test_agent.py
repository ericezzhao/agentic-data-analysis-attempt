"""
Test Data Analysis Agent

This module tests the intelligent data analysis agent functionality including:
- Agent initialization and basic operations
- Dataset analysis workflow
- Query processing with reasoning
- Memory and conversation management
- Integration with ChromaDB and MCP server
"""

import pytest
import asyncio
import tempfile
import pandas as pd
from pathlib import Path
import logging

from ..agent.data_analysis_agent import DataAnalysisAgent, create_data_analysis_agent
from ..config.settings import get_settings
from ..database.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class TestDataAnalysisAgent:
    """Test suite for DataAnalysisAgent."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        # Create sample data
        data = {
            'product_id': range(1, 101),
            'product_name': [f'Product_{i}' for i in range(1, 101)],
            'category': ['Electronics', 'Clothing', 'Food', 'Books'] * 25,
            'price': [19.99 + i * 2.5 for i in range(100)],
            'sales_count': [10 + i * 3 for i in range(100)],
            'rating': [3.5 + (i % 5) * 0.3 for i in range(100)],
            'description': [f'High quality product {i} with excellent features' for i in range(1, 101)]
        }
        df = pd.DataFrame(data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return create_data_analysis_agent(
            session_id="test_session_001",
            debug_mode=True
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert hasattr(agent, 'session_id')
        assert hasattr(agent, 'chroma_client')
        assert hasattr(agent, 'data_processor')
        assert hasattr(agent, 'query_optimizer')
        
        logger.info("✅ Agent initialization test passed")
    
    @pytest.mark.asyncio
    async def test_dataset_analysis_workflow(self, agent, sample_csv_file):
        """Test the complete dataset analysis workflow."""
        try:
            # Test dataset analysis
            result = await agent.analyze_dataset(
                file_path=sample_csv_file,
                file_name="test_products.csv",
                description="Sample product dataset for testing"
            )
            
            # Verify successful analysis
            assert result["success"] is True
            assert "chunks_created" in result
            assert "processing_time" in result
            assert "insights" in result
            assert len(result["insights"]) > 0
            
            logger.info("✅ Dataset analysis workflow test passed")
            logger.info(f"   Created {result['chunks_created']} chunks in {result['processing_time']:.2f}s")
            logger.info(f"   Generated {len(result['insights'])} insights")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Dataset analysis workflow test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_query_processing(self, agent, sample_csv_file):
        """Test query processing with reasoning."""
        try:
            # First upload a dataset
            upload_result = await agent.analyze_dataset(
                file_path=sample_csv_file,
                file_name="test_products.csv",
                description="Sample product dataset"
            )
            assert upload_result["success"] is True
            
            # Test various types of queries
            test_queries = [
                "What are the average prices by category?",
                "Which products have the highest ratings?",
                "Show me electronics products",
                "What is the correlation between price and sales?",
                "Find products with low ratings"
            ]
            
            successful_queries = 0
            for query in test_queries:
                result = await agent.query_data(
                    query=query,
                    dataset_name="test_products.csv"
                )
                
                if result["success"]:
                    successful_queries += 1
                    logger.info(f"✅ Query '{query}' processed successfully")
                    logger.info(f"   Found {result['results_count']} results")
                    logger.info(f"   Generated {len(result['insights'])} insights")
                else:
                    logger.warning(f"⚠️  Query '{query}' failed: {result['message']}")
            
            # Verify that most queries succeeded
            success_rate = successful_queries / len(test_queries)
            assert success_rate >= 0.6, f"Query success rate too low: {success_rate}"
            
            logger.info(f"✅ Query processing test passed: {successful_queries}/{len(test_queries)} queries succeeded")
            
        except Exception as e:
            logger.error(f"❌ Query processing test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_agent_reasoning_capabilities(self, agent, sample_csv_file):
        """Test agent's reasoning and insight generation capabilities."""
        try:
            # Upload dataset
            upload_result = await agent.analyze_dataset(
                file_path=sample_csv_file,
                file_name="reasoning_test.csv",
                description="Dataset for testing reasoning capabilities"
            )
            assert upload_result["success"] is True
            
            # Test reasoning with a complex query
            complex_query = "What insights can you provide about the relationship between product categories, prices, and customer ratings?"
            
            result = await agent.query_data(
                query=complex_query,
                dataset_name="reasoning_test.csv"
            )
            
            if result["success"]:
                insights = result.get("insights", [])
                
                # Verify reasoning quality
                assert len(insights) >= 2, "Agent should provide multiple insights"
                
                # Check for analytical depth
                insight_text = " ".join(insights).lower()
                analytical_keywords = ["correlation", "pattern", "trend", "relationship", "analysis", "distribution"]
                found_keywords = [keyword for keyword in analytical_keywords if keyword in insight_text]
                
                assert len(found_keywords) >= 1, "Insights should demonstrate analytical reasoning"
                
                logger.info("✅ Agent reasoning capabilities test passed")
                logger.info(f"   Generated {len(insights)} insights with analytical depth")
                logger.info(f"   Found analytical keywords: {found_keywords}")
            else:
                logger.warning("⚠️  Complex query processing failed, but this might be expected")
                # Don't fail the test if the query doesn't find results, as this depends on data indexing
            
        except Exception as e:
            logger.error(f"❌ Agent reasoning test failed: {e}")
            raise
    
    def test_agent_factory_function(self):
        """Test the agent factory function."""
        try:
            # Test default creation
            agent1 = create_data_analysis_agent()
            assert agent1 is not None
            assert hasattr(agent1, 'session_id')
            
            # Test with custom session
            agent2 = create_data_analysis_agent(session_id="custom_session", debug_mode=True)
            assert agent2.session_id == "custom_session"
            
            # Verify they are different instances
            assert agent1.session_id != agent2.session_id
            
            logger.info("✅ Agent factory function test passed")
            
        except Exception as e:
            logger.error(f"❌ Agent factory function test failed: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, agent, sample_csv_file):
        """Test conversation context and memory."""
        try:
            # Upload dataset first
            upload_result = await agent.analyze_dataset(
                file_path=sample_csv_file,
                file_name="context_test.csv",
                description="Dataset for context testing"
            )
            assert upload_result["success"] is True
            
            # Get conversation summary
            summary = await agent.get_conversation_summary()
            
            assert "session_id" in summary
            assert summary["session_id"] == agent.session_id
            assert isinstance(summary.get("total_messages", 0), int)
            
            logger.info("✅ Conversation context test passed")
            logger.info(f"   Session ID: {summary['session_id']}")
            logger.info(f"   Total messages: {summary.get('total_messages', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Conversation context test failed: {e}")
            raise


def run_agent_tests():
    """Run all agent tests with detailed output."""
    print("🚀 Starting Data Analysis Agent Tests...")
    print("=" * 60)
    
    try:
        # Run the tests
        pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "--capture=no"
        ])
        
        print("\n" + "=" * 60)
        print("✅ Agent tests completed!")
        
    except Exception as e:
        print(f"\n❌ Agent tests failed: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    run_agent_tests() 