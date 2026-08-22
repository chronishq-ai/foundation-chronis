"""
Sprint 20: Cross-User Isolation CI Check
Fails the build if any logic attempts to cross the PrivateIdentityGraph boundary.
"""
import sys
from src.frontier.identity_graph import PrivateIdentityGraph

def test_cross_user_isolation():
    graph_alice = PrivateIdentityGraph("user_alice")
    graph_bob = PrivateIdentityGraph("user_bob")
    
    try:
        # Attempt to merge graphs (simulating a global aggregation bug)
        graph_alice.merge_graphs(graph_bob)
    except PermissionError as e:
        print("PASS: Cross-user isolation enforced.")
        return True
        
    print("FAIL: Cross-user isolation violated!")
    return False

if __name__ == "__main__":
    success = test_cross_user_isolation()
    sys.exit(0 if success else 1)
