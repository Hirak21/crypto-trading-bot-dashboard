#!/usr/bin/env python3
"""
Test runner for the crypto trading bot test suite.

This script runs all unit tests, integration tests, and generates
a comprehensive test report.
"""

import unittest
import sys
import os
import asyncio
import time
from typing import List, Dict, Any
from io import StringIO
import importlib.util

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, status: str, duration: float, error: str = None):
        self.name = name
        self.status = status  # 'PASS', 'FAIL', 'ERROR', 'SKIP'
        self.duration = duration
        self.error = error


class TestSuiteRunner:
    """Comprehensive test suite runner."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("🚀 Starting Crypto Trading Bot Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test modules to run
        test_modules = [
            'tests.test_mock_data',
            'tests.test_strategies',
            'tests.test_technical_analysis',
            'tests.test_managers',
            'tests.test_error_recovery',
            'tests.test_end_to_end_integration'
        ]
        
        # Run each test module
        for module_name in test_modules:
            print(f"\n📋 Running {module_name}")
            print("-" * 40)
            
            try:
                self._run_test_module(module_name)
            except Exception as e:
                print(f"❌ Failed to run {module_name}: {e}")
                self.results.append(TestResult(
                    name=module_name,
                    status='ERROR',
                    duration=0.0,
                    error=str(e)
                ))
        
        total_duration = time.time() - start_time
        
        # Generate summary
        self._generate_summary(total_duration)
        
        return {
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'errors': self.error_tests,
            'skipped': self.skipped_tests,
            'duration': total_duration,
            'results': self.results
        }
    
    def _run_test_module(self, module_name: str):
        """Run tests for a specific module."""
        try:
            # Import the test module
            module = importlib.import_module(module_name)
            
            # Check if module has async tests
            if hasattr(module, 'AsyncTestRunner'):
                self._run_async_test_module(module)
            else:
                self._run_sync_test_module(module)
                
        except ImportError as e:
            print(f"⚠️  Could not import {module_name}: {e}")
            self.results.append(TestResult(
                name=module_name,
                status='ERROR',
                duration=0.0,
                error=f"Import error: {e}"
            ))
    
    def _run_sync_test_module(self, module):
        """Run synchronous tests from a module."""
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Capture test output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        
        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time
        
        # Process results
        self.total_tests += result.testsRun
        self.passed_tests += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        self.failed_tests += len(result.failures)
        self.error_tests += len(result.errors)
        self.skipped_tests += len(result.skipped)
        
        # Print results
        if result.wasSuccessful():
            print(f"✅ All tests passed ({result.testsRun} tests, {duration:.2f}s)")
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            
            # Print failure details
            for test, traceback in result.failures:
                print(f"   FAIL: {test}")
                print(f"   {traceback.split('AssertionError:')[-1].strip()}")
            
            for test, traceback in result.errors:
                print(f"   ERROR: {test}")
                print(f"   {traceback.split('Exception:')[-1].strip()}")
    
    def _run_async_test_module(self, module):
        """Run asynchronous tests from a module."""
        print("🔄 Running async tests...")
        
        try:
            # Get the async test runner
            async_runner = module.AsyncTestRunner()
            
            # Run async tests
            if hasattr(async_runner, 'run_tests'):
                async_runner.run_tests()
            elif hasattr(async_runner, 'run_async_tests'):
                async_runner.run_async_tests()
            else:
                print("⚠️  No recognized async test runner method found")
                
        except Exception as e:
            print(f"❌ Error running async tests: {e}")
            self.results.append(TestResult(
                name=f"{module.__name__}_async",
                status='ERROR',
                duration=0.0,
                error=str(e)
            ))
    
    def _generate_summary(self, total_duration: float):
        """Generate and print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests:    {self.total_tests}")
        print(f"✅ Passed:      {self.passed_tests}")
        print(f"❌ Failed:      {self.failed_tests}")
        print(f"💥 Errors:      {self.error_tests}")
        print(f"⏭️  Skipped:     {self.skipped_tests}")
        print(f"⏱️  Duration:    {total_duration:.2f}s")
        
        # Calculate success rate
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Overall status
        if self.failed_tests == 0 and self.error_tests == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n💔 {self.failed_tests + self.error_tests} TESTS FAILED")
            return False


class ComponentTestRunner:
    """Run tests for specific components."""
    
    @staticmethod
    def run_strategy_tests():
        """Run only strategy-related tests."""
        print("🎯 Running Strategy Tests Only")
        suite = unittest.TestLoader().loadTestsFromName('tests.test_strategies')
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)
    
    @staticmethod
    def run_technical_analysis_tests():
        """Run only technical analysis tests."""
        print("📈 Running Technical Analysis Tests Only")
        suite = unittest.TestLoader().loadTestsFromName('tests.test_technical_analysis')
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)
    
    @staticmethod
    def run_manager_tests():
        """Run only manager tests."""
        print("👔 Running Manager Tests Only")
        suite = unittest.TestLoader().loadTestsFromName('tests.test_managers')
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)
    
    @staticmethod
    def run_integration_tests():
        """Run only integration tests."""
        print("🔗 Running Integration Tests Only")
        suite = unittest.TestLoader().loadTestsFromName('tests.test_end_to_end_integration')
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Test Runner')
    parser.add_argument('--component', choices=[
        'strategies', 'technical', 'managers', 'integration', 'all'
    ], default='all', help='Run tests for specific component')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage analysis (requires coverage.py)')
    
    args = parser.parse_args()
    
    # Set up coverage if requested
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
            print("📊 Running with coverage analysis...")
        except ImportError:
            print("⚠️  Coverage analysis requested but coverage.py not installed")
            print("   Install with: pip install coverage")
            args.coverage = False
    
    # Run tests based on component selection
    if args.component == 'all':
        runner = TestSuiteRunner()
        results = runner.run_all_tests()
        success = results['failed'] == 0 and results['errors'] == 0
    else:
        component_runner = ComponentTestRunner()
        
        if args.component == 'strategies':
            result = component_runner.run_strategy_tests()
        elif args.component == 'technical':
            result = component_runner.run_technical_analysis_tests()
        elif args.component == 'managers':
            result = component_runner.run_manager_tests()
        elif args.component == 'integration':
            result = component_runner.run_integration_tests()
        
        success = result.wasSuccessful()
    
    # Generate coverage report if requested
    if args.coverage:
        try:
            cov.stop()
            cov.save()
            
            print("\n📊 COVERAGE REPORT")
            print("=" * 40)
            cov.report()
            
            # Generate HTML coverage report
            cov.html_report(directory='htmlcov')
            print("📄 HTML coverage report generated in 'htmlcov/' directory")
            
        except Exception as e:
            print(f"⚠️  Error generating coverage report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()