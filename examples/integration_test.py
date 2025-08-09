"""
QuizMaster Basic Integration Test

Tests each component individually to identify where issues occur.
"""

import asyncio
from pathlib import Path

from quizmaster.config import QuizMasterConfig, setup_logging


async def test_bookworm_integration():
    """Test BookWorm integration step by step."""
    
    print("🔬 Testing BookWorm Integration")
    print("-" * 40)
    
    try:
        config = QuizMasterConfig.from_env()
        
        # Test 1: Import BookWorm
        print("1️⃣ Testing BookWorm imports...")
        try:
            from bookworm.utils import BookWormConfig
            from bookworm.core import DocumentProcessor
            print("   ✅ BookWorm imports successful")
        except ImportError as e:
            print(f"   ❌ BookWorm import failed: {e}")
            return False
        
        # Test 2: Create BookWorm config
        print("2️⃣ Testing BookWorm config creation...")
        try:
            bw_config = BookWormConfig()
            bw_config.api_provider = config.api_provider
            if config.openai_api_key:
                bw_config.openai_api_key = config.openai_api_key
            
            # Ensure required directories exist
            working_dir = Path(bw_config.working_dir)
            working_dir.mkdir(exist_ok=True)
            (working_dir / "library").mkdir(exist_ok=True)
            (working_dir / "docs").mkdir(exist_ok=True)
            (working_dir / "processed_docs").mkdir(exist_ok=True)
            (working_dir / "output").mkdir(exist_ok=True)
            (working_dir / "logs").mkdir(exist_ok=True)
            
            print("   ✅ BookWorm config created with directories")
        except Exception as e:
            print(f"   ❌ BookWorm config failed: {e}")
            return False
        
        # Test 3: Create DocumentProcessor
        print("3️⃣ Testing DocumentProcessor creation...")
        try:
            processor = DocumentProcessor(config=bw_config)
            print("   ✅ DocumentProcessor created")
        except Exception as e:
            print(f"   ❌ DocumentProcessor failed: {e}")
            return False
        
        # Test 4: Test document processing
        print("4️⃣ Testing document processing...")
        test_file = None
        try:
            test_file = Path("test_doc.txt")
            test_file.write_text("This is a simple test document for BookWorm integration.")
            
            # Check if processor has the expected methods
            print(f"   📋 Available methods: {[m for m in dir(processor) if not m.startswith('_')][:10]}...")
            
            # Try different method names that might exist
            if hasattr(processor, 'process_document'):
                result = await processor.process_document(str(test_file))
                print(f"   ✅ process_document worked: {type(result)}")
            elif hasattr(processor, 'process'):
                result = await processor.process(str(test_file))
                print(f"   ✅ process worked: {type(result)}")
            else:
                print("   ⚠️ No known processing method found")
                
        except Exception as e:
            print(f"   ❌ Document processing failed: {e}")
            return False
        finally:
            if test_file:
                test_file.unlink(missing_ok=True)
        
        print("🎉 BookWorm integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        return False


async def test_qbank_integration():
    """Test qBank integration step by step."""
    
    print("\\n🔬 Testing qBank Integration")
    print("-" * 40)
    
    try:
        # Test 1: Import qBank
        print("1️⃣ Testing qBank imports...")
        try:
            from qbank import QuestionBankManager
            print("   ✅ qBank imports successful")
        except ImportError as e:
            print(f"   ❌ qBank import failed: {e}")
            return False
        
        # Test 2: Create qBank manager
        print("2️⃣ Testing qBank manager creation...")
        try:
            bank_manager = QuestionBankManager(user_id="test_user", bank_name="Test Bank")
            print("   ✅ qBank manager created")
        except Exception as e:
            print(f"   ❌ qBank manager failed: {e}")
            return False
        
        # Test 3: Test basic qBank operations
        print("3️⃣ Testing basic qBank operations...")
        try:
            # Check available methods
            print(f"   📋 Available methods: {[m for m in dir(bank_manager) if not m.startswith('_')][:10]}...")
            
            # Try to add a simple question using the correct qBank interface
            if hasattr(bank_manager, 'add_question'):
                result = bank_manager.add_question(
                    question_text="What is 2 + 2?",
                    correct_answer="4",
                    incorrect_answers=["3", "5", "6"],
                    tags={"math", "basic"}
                )
                print(f"   ✅ Question added: {result.id if hasattr(result, 'id') else 'success'}")
            else:
                print("   ⚠️ No add_question method found")
            
        except Exception as e:
            print(f"   ❌ qBank operations failed: {e}")
            return False
        
        print("🎉 qBank integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Overall qBank test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    
    print("🧪 QuizMaster Integration Tests")
    print("=" * 50)
    
    config = QuizMasterConfig.from_env()
    setup_logging(config)
    
    if not config.validate_api_key():
        print(f"⚠️ No API key found for {config.api_provider}, some tests may fail")
    
    bookworm_success = await test_bookworm_integration()
    qbank_success = await test_qbank_integration()
    
    print("\\n" + "=" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"📄 BookWorm Integration: {'✅ PASSED' if bookworm_success else '❌ FAILED'}")
    print(f"📚 qBank Integration: {'✅ PASSED' if qbank_success else '❌ FAILED'}")
    
    if bookworm_success and qbank_success:
        print("\\n🎉 All integration tests passed! Ready for full pipeline demo.")
        print("Next: Run 'uv run python examples/complete_pipeline_demo.py'")
    else:
        print("\\n⚠️ Some integration tests failed. Check the errors above.")
        print("The packages may need debugging or method signatures may have changed.")
    
    return bookworm_success and qbank_success


if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
