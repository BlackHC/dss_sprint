"""Console script for dss_sprint."""

import fire


def help():
    print("dss_sprint")
    print("=" * len("dss_sprint"))
    print("Data Subset Selection Sprint")

def main():
    fire.Fire({
        "help": help
    })


if __name__ == "__main__":
    main() # pragma: no cover
