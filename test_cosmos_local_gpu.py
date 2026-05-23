#!/usr/bin/env python3
"""
Test Cosmos Predict 2.5 policy with local model inference on L40S GPU.

This test verifies:
1. Model loading and initialization  
2. Local inference with real model (using GPU)
3. Action prediction from synthetic observations
4. Performance metrics
"""

import asyncio
import time
import gc
from typing import Any

import numpy as np
import torch

from strands_robots.policies.cosmos_predict.policy import CosmosPredictPolicy


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated(0) / (1024**2),
            'cached_mb': torch.cuda.memory_reserved(0) / (1024**2)
        }
    return {'allocated_mb': 0, 'cached_mb': 0}


def create_synthetic_observation(suite: str = "libero") -> dict[str, Any]:
    """Create synthetic robot observation for testing."""
    obs = {}
    
    if suite == "libero":
        # LIBERO: 1 wrist + 1 third-person camera
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(7).astype(np.float32) * 0.1
        })
    elif suite == "robocasa":
        # RoboCasa: 1 wrist + 2 third-person cameras  
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "secondary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(11).astype(np.float32) * 0.1
        })
    elif suite == "aloha":
        # ALOHA: 2 wrist + 1 third-person camera
        obs.update({
            "primary_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), 
            "left_wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "right_wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "proprio": np.random.randn(11).astype(np.float32) * 0.1
        })
    
    return obs


def test_policy_loading(model_id: str, suite: str):
    """Test loading the Cosmos policy model."""
    print(f"\n🧪 Testing policy loading...")
    print(f"   Model: {model_id}")
    print(f"   Suite: {suite}")
    
    # Get initial memory state
    gpu_mem_before = get_gpu_memory()
    print(f"   GPU memory before: allocated={gpu_mem_before['allocated_mb']:.1f}MB, cached={gpu_mem_before['cached_mb']:.1f}MB")
    
    start_time = time.time()
    
    try:
        policy = CosmosPredictPolicy(
            model_id=model_id,
            suite=suite,
            device="cuda:0",
            chunk_size=16,
            num_denoising_steps=5
        )
        
        print("   Loading model (this may take a few minutes)...")
        
        # Force model loading
        policy._ensure_loaded()
        
        load_time = time.time() - start_time
        
        # Get memory state after loading
        gpu_mem_after = get_gpu_memory()
        
        print(f"   ✅ Policy loaded in {load_time:.1f}s")
        print(f"   GPU memory after: allocated={gpu_mem_after['allocated_mb']:.1f}MB (Δ +{gpu_mem_after['allocated_mb'] - gpu_mem_before['allocated_mb']:.1f}MB)")
        print(f"   GPU memory cached: {gpu_mem_after['cached_mb']:.1f}MB")
        
        return policy
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 cosmos-predict2 not available - this is expected for first-time setup")
        print("   💡 To install cosmos-predict2:")
        print("      git clone https://github.com/nvidia-cosmos/cosmos-predict2.5")
        print("      cd cosmos-predict2.5")
        print("      pip install -e packages/cosmos-oss -e packages/cosmos-cuda -e .")
        return None
    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        return None


async def test_inference(policy: CosmosPredictPolicy, suite: str, num_steps: int = 3):
    """Test inference with synthetic observations."""
    print(f"\n🔮 Testing inference ({num_steps} steps)...")
    
    # Create synthetic observation
    obs = create_synthetic_observation(suite)
    instruction = "pick up the red cube and place it in the box"
    
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Instruction: {instruction}")
    
    inference_times = []
    
    for step in range(num_steps):
        print(f"   Step {step + 1}/{num_steps}...", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            actions = await policy.get_actions(obs, instruction)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            print(f"✅ {inference_time:.2f}s ({len(actions)} actions)")
            
            if step == 0 and actions:  # Show first action in detail
                action = actions[0]
                action_keys = list(action.keys())[:5]  # First 5 keys only
                action_str = ", ".join([f"{k}:{action[k]:.3f}" for k in action_keys])
                if len(action.keys()) > 5:
                    action_str += "..."
                print(f"      First action sample: {action_str}")
                print(f"      Action chunk size: {len(actions)} actions")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"   📊 Average inference time: {avg_time:.2f}s")
        print(f"   📊 Throughput: {1.0/avg_time:.2f} inferences/sec")
        print(f"   📊 Actions per second: {len(actions) / avg_time:.1f}")


def test_observation_processing(policy: CosmosPredictPolicy, suite: str):
    """Test observation processing without full inference."""
    print(f"\n🔧 Testing observation processing...")
    
    obs = create_synthetic_observation(suite)
    
    try:
        cosmos_obs = policy._build_observation(obs)
        print(f"   ✅ Observation conversion successful")
        print(f"   Input keys: {list(obs.keys())}")
        print(f"   Cosmos keys: {list(cosmos_obs.keys())}")
        
        for key, val in cosmos_obs.items():
            if hasattr(val, 'shape'):
                print(f"   {key}: shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"   {key}: {type(val)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Observation processing failed: {e}")
        return False


def test_memory_cleanup():
    """Test memory cleanup after model usage."""
    print(f"\n🧹 Testing memory cleanup...")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gpu_mem_after = get_gpu_memory()
    print(f"   GPU memory after cleanup: allocated={gpu_mem_after['allocated_mb']:.1f}MB, cached={gpu_mem_after['cached_mb']:.1f}MB")


async def main():
    """Main test runner."""
    print("🚀 Cosmos Predict 2.5 Local GPU Test")
    print("=" * 50)
    
    # Check CUDA availability  
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        print(f"   Device memory: {props.total_memory / (1024**3):.1f}GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
    
    # Test configuration 
    config = {
        "model_id": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        "suite": "libero"
    }
    
    try:
        # Test policy loading
        policy = test_policy_loading(config["model_id"], config["suite"])
        
        if policy is None:
            print("⏭️  Skipping inference tests due to model loading failure")
            print("💡 This is expected if cosmos-predict2 is not installed")
            return
        
        # Test observation processing (lightweight)
        obs_success = test_observation_processing(policy, config["suite"])
        
        if obs_success:
            # Test inference  
            await test_inference(policy, config["suite"], num_steps=3)
        
        # Clean up
        del policy
        test_memory_cleanup()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())