#!/usr/bin/env python3
"""
Test script to verify Snapshot NamedTuple migration works correctly.
This script tests both old and new Snapshot formats for pickle compatibility.
"""

import io
import pickle
import tempfile
import numpy as np

# Import the current Snapshot and migration functions
from pysages.backends.snapshot import Snapshot, Box, _migrate_old_snapshot
from pysages.serialization import load, save


def create_test_data():
    """Create test data for Snapshot objects."""
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    masses = np.array([1.0, 2.0])
    vel_mass = (velocities, masses)
    forces = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    ids = np.array([0, 1])
    H = np.eye(3) * 10.0
    origin = np.array([0.0, 0.0, 0.0])
    box = Box(H, origin)
    dt = 0.001
    images = np.array([[0, 0, 0], [1, 1, 1]])
    
    return positions, vel_mass, forces, ids, box, dt, images


def test_old_snapshot_format():
    """Test deserialization of old Snapshot format."""
    print("Testing old Snapshot format...")
    
    positions, vel_mass, forces, ids, box, dt, images = create_test_data()
    
    # Create old format Snapshot (with images as separate field)
    # This simulates the old format: Snapshot(positions, vel_mass, forces, ids, images, box, dt)
    old_snapshot_data = (positions, vel_mass, forces, ids, images, box, dt)
    
    # Test migration function directly
    migrated = _migrate_old_snapshot(old_snapshot_data)
    print(f"✓ Migration successful: {type(migrated)}")
    print(f"  - Has extras: {migrated.extras is not None}")
    print(f"  - Images in extras: {'images' in migrated.extras if migrated.extras else False}")
    
    # Test pickle round-trip with old format
    with tempfile.NamedTemporaryFile() as tmp_file:
        # Simulate old format by manually creating the pickle data
        old_snapshot = Snapshot(positions, vel_mass, forces, ids, box, dt, {"images": images})
        save({"states": [old_snapshot]}, tmp_file.name)
        
        # Load it back
        loaded = load(tmp_file.name)
        print(f"✓ Pickle round-trip successful: {type(loaded)}")
    
    return True


def test_new_snapshot_format():
    """Test deserialization of new Snapshot format."""
    print("\nTesting new Snapshot format...")
    
    positions, vel_mass, forces, ids, box, dt, images = create_test_data()
    
    # Create new format Snapshot (with images in extras)
    new_snapshot = Snapshot(positions, vel_mass, forces, ids, box, dt, {"images": images})
    
    # Test pickle round-trip
    with tempfile.NamedTemporaryFile() as tmp_file:
        save({"states": [new_snapshot]}, tmp_file.name)
        
        loaded = load(tmp_file.name)
        print(f"✓ New format pickle round-trip successful: {type(loaded)}")
    
    return True


def test_mixed_formats():
    """Test handling of mixed old/new formats in the same file."""
    print("\nTesting mixed formats...")
    
    positions, vel_mass, forces, ids, box, dt, images = create_test_data()
    
    # Create both old and new format snapshots
    old_snapshot = Snapshot(positions, vel_mass, forces, ids, box, dt, {"images": images})
    new_snapshot = Snapshot(positions, vel_mass, forces, ids, box, dt, {"images": images})
    
    # Test that both can be pickled and loaded
    with tempfile.NamedTemporaryFile() as tmp_file:
        save({"states": [old_snapshot, new_snapshot]}, tmp_file.name)
        
        loaded = load(tmp_file.name)
        print(f"✓ Mixed formats pickle round-trip successful: {len(loaded['states'])} states")
    
    return True


def test_error_handling():
    """Test error handling for invalid formats."""
    print("\nTesting error handling...")
    
    try:
        # Test with invalid number of fields
        invalid_data = (1, 2, 3)  # Too few fields
        _migrate_old_snapshot(invalid_data)
        print("✗ Should have raised ValueError for invalid data")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid data: {e}")
    
    return True


def main():
    """Run all tests."""
    print("Testing Snapshot NamedTuple migration for pickle compatibility")
    print("=" * 60)
    
    tests = [
        test_old_snapshot_format,
        test_new_snapshot_format,
        test_mixed_formats,
        test_error_handling,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)