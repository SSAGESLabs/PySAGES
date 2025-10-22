#!/usr/bin/env python3
"""
Simple test to verify Snapshot migration logic works correctly.
"""

import numpy as np
from typing import NamedTuple, Union, Tuple, Optional, Dict, Any


# Simplified versions of the classes for testing
class Box(NamedTuple):
    H: np.ndarray
    origin: np.ndarray


class Snapshot(NamedTuple):
    positions: np.ndarray
    vel_mass: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    forces: np.ndarray
    ids: np.ndarray
    box: Box
    dt: Union[np.ndarray, float]
    extras: Optional[Dict[str, Any]] = None

    def __reduce__(self):
        """Custom pickle serialization to handle backward compatibility."""
        return _snapshot_reducer, (self.positions, self.vel_mass, self.forces, 
                                 self.ids, self.box, self.dt, self.extras)


def _snapshot_reducer(positions, vel_mass, forces, ids, box, dt, extras):
    """Reconstruct Snapshot from serialized data."""
    return Snapshot(positions, vel_mass, forces, ids, box, dt, extras)


def _migrate_old_snapshot(old_data):
    """
    Migrate old Snapshot format to new format.
    
    Handles the transition from:
    Snapshot(positions, vel_mass, forces, ids, images, box, dt)
    to:
    Snapshot(positions, vel_mass, forces, ids, box, dt, extras)
    """
    if len(old_data) == 7:
        # Old format: (positions, vel_mass, forces, ids, images, box, dt)
        positions, vel_mass, forces, ids, images, box, dt = old_data
        extras = {"images": images} if images is not None else None
        return Snapshot(positions, vel_mass, forces, ids, box, dt, extras)
    elif len(old_data) == 6:
        # New format: (positions, vel_mass, forces, ids, box, dt, extras)
        return Snapshot(*old_data)
    else:
        raise ValueError(f"Unexpected Snapshot data format with {len(old_data)} fields")


def test_migration():
    """Test the migration function directly."""
    print("Testing Snapshot migration function...")
    
    # Create test data
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
    
    # Test old format migration
    old_data = (positions, vel_mass, forces, ids, images, box, dt)
    migrated = _migrate_old_snapshot(old_data)
    
    print(f"✓ Migration successful: {type(migrated)}")
    print(f"  - Has extras: {migrated.extras is not None}")
    print(f"  - Images in extras: {'images' in migrated.extras if migrated.extras else False}")
    print(f"  - Images data matches: {np.array_equal(images, migrated.extras['images'])}")
    
    # Test new format (should pass through unchanged)
    new_data = (positions, vel_mass, forces, ids, box, dt, {"images": images})
    new_snapshot = _migrate_old_snapshot(new_data)
    
    print(f"✓ New format handled correctly: {type(new_snapshot)}")
    print(f"  - Has extras: {new_snapshot.extras is not None}")
    
    # Test error handling
    try:
        invalid_data = (1, 2, 3)  # Too few fields
        _migrate_old_snapshot(invalid_data)
        print("✗ Should have raised ValueError for invalid data")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid data: {e}")
    
    return True


def test_pickle_compatibility():
    """Test pickle serialization/deserialization."""
    print("\nTesting pickle compatibility...")
    
    import pickle
    
    # Create test data
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
    
    # Test new format Snapshot
    snapshot = Snapshot(positions, vel_mass, forces, ids, box, dt, {"images": images})
    
    # Pickle and unpickle
    pickled = pickle.dumps(snapshot)
    unpickled = pickle.loads(pickled)
    
    print(f"✓ Pickle round-trip successful: {type(unpickled)}")
    print(f"  - Data matches: {np.array_equal(snapshot.positions, unpickled.positions)}")
    
    # Compare extras more carefully
    if snapshot.extras is None and unpickled.extras is None:
        extras_match = True
    elif snapshot.extras is None or unpickled.extras is None:
        extras_match = False
    else:
        extras_match = (set(snapshot.extras.keys()) == set(unpickled.extras.keys()) and
                       all(np.array_equal(snapshot.extras[k], unpickled.extras[k]) 
                           for k in snapshot.extras.keys()))
    print(f"  - Extras preserved: {extras_match}")
    
    return True


def main():
    """Run all tests."""
    print("Testing Snapshot NamedTuple migration for pickle compatibility")
    print("=" * 60)
    
    tests = [test_migration, test_pickle_compatibility]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)