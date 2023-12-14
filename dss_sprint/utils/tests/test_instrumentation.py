from dss_sprint.utils.instrumentation import Context, Instrumentation


def test_instrumentation_example():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.record("loss", 1)

        with instrument.scope("my_scope"):
            instrument.record("info", 2)

    assert run_info == Context(loss=[1], my_scope=[Context(info=[2])])


def test_instrumentation_decorator():
    instrument = Instrumentation()

    @instrument.scope("my_scope")
    def my_func():
        instrument.record("info", 2)

    with instrument.collect() as run_info:
        instrument.record("loss", 1)
        my_func()

    assert run_info == Context(loss=[1], my_scope=[Context(info=[2])])


def test_instrumentation_nested_scopes():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.record("loss", 1)

        with instrument.scope("my_scope"):
            instrument.record("info", 2)

            with instrument.scope("my_scope"):
                instrument.record("info", 3)

    assert run_info == Context(
        loss=[1],
        my_scope=[
            Context(
                info=[2],
                my_scope=[
                    Context(info=[3]),
                ],
            ),
        ],
    )


def test_instrumentation_nested_collects():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.record("loss", 1)

        with instrument.collect() as run_info_inner:
            instrument.record("loss", 2)

            with instrument.scope("my_scope"):
                instrument.record("info", 3)

    assert run_info == Context(
        loss=[1, 2],
        my_scope=[
            Context(info=[3]),
        ],
    )

    assert run_info_inner == Context(
        loss=[2],
        my_scope=[
            Context(info=[3]),
        ],
    )


def test_instrumentation_replay():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.record("loss", 1)

        with instrument.scope("my_scope"):
            instrument.record("info", 2)

    assert list(instrument.replay(run_info)) == [
        ("loss", 1),
        ("my_scope", Context(info=[2])),
    ]


def test_instrumentation_deep_replay():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.record("loss", 1)

        with instrument.scope("my_scope"):
            instrument.record("info", 2)

    assert list(instrument.deep_replay(run_info)) == [
        (("loss",), 1),
        (("my_scope",), Context(info=[2])),
        (("my_scope", "info"), 2),
    ]

    # Check that we can pattern match as well
    for item in instrument.deep_replay(run_info):
        match item:
            case (["loss"], value):
                assert value == 1
            case (["my_scope"], context):
                assert context == Context(info=[2])
            case (["my_scope", "info"], value):
                assert value == 2


if __name__ == "__main__":
    test_instrumentation_example()
    test_instrumentation_decorator()
    test_instrumentation_nested_scopes()
    test_instrumentation_nested_collects()
    test_instrumentation_replay()
    test_instrumentation_deep_replay()
