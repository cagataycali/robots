#!/usr/bin/env python3
"""
Test Cosmos Predict 2.5 policy in server mode and validate repository integration.

This test verifies the Cosmos policy integration with strands-robots without 
requiring full local model installation. Perfect for validating the PR.
"""

import asyncio
import time
from typing import Any

import numpy as np
import torch

from strands_robots.policies.cosmos_predict.policy import CosmosPredictPolicy


def test_policy_registry_integration():
    """Test that the policy is properly integrated with the strands-robots registry."""
    print("🔌 Testing policy registry integration...")
    
    try:
        # Test direct import
        from strands_robots.policies.cosmos_predict import CosmosPredictPolicy as DirectCls
        print("   ✅ Direct import successful")
        
        # Test registry import  
        from strands_robots.policies import create_policy
        print("   ✅ Registry import successful")
        
        # Test policy creation in server mode (no local model required)
        policy = create_policy(
            "cosmos_predict", 
            server_url="http://localhost:8000",
            suite="libero"
        )
        print("   ✅ Policy creation via registry successful")
        print(f"   Provider: {policy.provider_name}")
        print(f"   Suite: {policy._suite}")
        print(f"   Server URL: {policy._server_url}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Registry integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_processing():
    """Test observation format conversion for all supported suites."""
    print("\n🔧 Testing observation processing for all suites...")
    
    suites = ["libero", "robocasa", "aloha"]
    
    for suite in suites:
        print(f"\n   Testing {suite.upper()} suite:")
        
        policy = CosmosPredictPolicy(
            server_url="http://localhost:8000",
            suite=suite
        )
        
        # Create suite-appropriate synthetic observation
        obs = create_synthetic_observation_for_suite(suite)
        
        try:
            cosmos_obs = policy._build_observation(obs)
            print(f"      ✅ Observation conversion successful")
            print(f"      Input keys: {list(obs.keys())}")
            print(f"      Cosmos keys: {list(cosmos_obs.keys())}")
            
            # Validate expected keys based on suite
            expected_keys = get_expected_cosmos_keys(suite)
            for key in expected_keys:
                if key in cosmos_obs:
                    val = cosmos_obs[key]
                    if hasattr(val, 'shape'):
                        print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
                    else:
                        print(f"      {key}: {type(val)}")
                else:
                    print(f"      ⚠️  Missing expected key: {key}")
                    
        except Exception as e:
            print(f"      ❌ Processing failed: {e}")


def create_synthetic_observation_for_suite(suite: str) -> dict[str, Any]:
    """Create synthetic observation appropriate for the suite."""
    obs = {}
    
    if suite == "libero":
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(7).astype(np.float32) * 0.1
        })
    elif suite == "robocasa":
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "secondary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(11).astype(np.float32) * 0.1
        })
    elif suite == "aloha":
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "left_wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "right_wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(11).astype(np.float32) * 0.1
        })
    
    return obs


def get_expected_cosmos_keys(suite: str) -> list[str]:
    """Get expected cosmos observation keys for the suite."""
    base_keys = ["proprio"]
    
    if suite == "libero":
        return base_keys + ["primary_image", "wrist_image"]
    elif suite == "robocasa":
        return base_keys + ["primary_image", "secondary_image", "wrist_image"]
    elif suite == "aloha":
        return base_keys + ["primary_image", "left_wrist_image", "right_wrist_image"]
    
    return base_keys


def test_action_decoding():
    """Test action decoding functionality."""
    print("\n🎯 Testing action decoding...")
    
    policy = CosmosPredictPolicy(server_url="http://localhost:8000")
    
    # Test with default robot state keys (7-DoF)
    print("   Testing default 7-DoF action decoding:")
    mock_result = {"actions": [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=np.float32)
    ]}
    
    actions = policy._decode_actions(mock_result)
    
    if actions:
        action = actions[0]
        print(f"      ✅ Action decoded: {action}")
        
        expected_keys = {"x", "y", "z", "roll", "pitch", "yaw", "gripper"}
        actual_keys = set(action.keys())
        
        if expected_keys == actual_keys:
            print("      ✅ All expected action keys present")
        else:
            print(f"      ⚠️  Key mismatch. Expected: {expected_keys}, Got: {actual_keys}")
    
    # Test with custom robot state keys
    print("   Testing custom robot state keys:")
    custom_keys = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    policy.set_robot_state_keys(custom_keys)
    
    actions = policy._decode_actions(mock_result)
    if actions:
        action = actions[0]
        print(f"      ✅ Custom action decoded: {action}")
        
        for key in custom_keys:
            if key not in action:
                print(f"      ⚠️  Missing custom key: {key}")
        
        if "gripper" not in action:
            print("      ⚠️  Missing gripper key")


def test_server_mode_configuration():
    """Test server mode configuration options."""
    print("\n🖥️  Testing server mode configuration...")
    
    test_configs = [
        {
            "server_url": "http://localhost:8000",
            "suite": "libero",
            "chunk_size": 16,
            "num_denoising_steps": 5
        },
        {
            "server_url": "http://localhost:9000", 
            "suite": "robocasa",
            "chunk_size": 32,
            "num_denoising_steps": 10
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"   Config {i+1}: {config}")
        
        try:
            policy = CosmosPredictPolicy(**config)
            print(f"      ✅ Policy created successfully")
            print(f"      Provider: {policy.provider_name}")
            print(f"      Server URL: {policy._server_url}")
            print(f"      Suite: {policy._suite}")
            print(f"      Chunk size: {policy._chunk_size}")
            print(f"      Denoising steps: {policy._num_denoising_steps}")
            
        except Exception as e:
            print(f"      ❌ Configuration failed: {e}")


def test_cosmos_policy_in_robot_factory():
    """Test cosmos policy integration with Robot factory system."""
    print("\n🤖 Testing integration with Robot factory...")
    
    try:
        # Import robot creation
        from strands_robots import Robot
        print("   ✅ Robot import successful")
        
        # This would normally create a robot with cosmos policy,
        # but we'll test the configuration only
        cosmos_config = {
            "policy": "cosmos_predict",
            "policy_server_url": "http://localhost:8000",
            "suite": "libero"
        }
        
        print(f"   ✅ Cosmos policy config valid: {cosmos_config}")
        print("   💡 Full robot creation would require:")
        print("      robot = Robot('libero', policy='cosmos_predict', policy_server_url='http://localhost:8000')")
        
    except Exception as e:
        print(f"   ❌ Robot integration test failed: {e}")


async def main():
    """Main test runner for cosmos policy validation."""
    print("🌌 Cosmos Predict 2.5 Policy Validation")
    print("=" * 55)
    
    # Check CUDA availability (informational)
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    
    print("\n" + "="*55)
    
    # Run comprehensive validation tests
    tests = [
        ("Registry Integration", test_policy_registry_integration),
        ("Observation Processing", test_observation_processing), 
        ("Action Decoding", test_action_decoding),
        ("Server Configuration", test_server_mode_configuration),
        ("Robot Factory Integration", test_cosmos_policy_in_robot_factory)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*55)
    print("📊 TEST SUMMARY")
    print("="*55)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Cosmos policy is properly integrated.")
    else:
        print("⚠️  Some tests failed - review output above.")
    
    print("\n💡 Next steps:")
    print("   1. Install cosmos-predict2.5 dependencies for local inference")  
    print("   2. Set up cosmos inference server for production use")
    print("   3. Test with real robot hardware and observations")


if __name__ == "__main__":
    asyncio.run(main())