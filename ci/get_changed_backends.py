import argparse

from merlin.models.utils.ci_utils import backend_has_changed, get_changed_backends

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=False, help="specific backend to check for changes"
    )
    args = parser.parse_args()

    if args.backend:
        print(str(backend_has_changed(args.backend)).lower())
    else:
        print(" ".join(get_changed_backends()))
