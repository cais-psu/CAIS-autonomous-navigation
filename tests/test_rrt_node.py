from autonomous_navigation.core.planners.rrt_node import RRTNode


def test_rrt_node():

    node = RRTNode(
        x=5.0,
        y=3.0,
        parent=1
    )

    assert node.x == 5.0
    assert node.y == 3.0
    assert node.parent == 1