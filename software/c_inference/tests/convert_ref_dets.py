import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [x.strip() for x in open(args.names)]

    with open(args.ref) as src, open(args.out, "w") as dst:
        dst.write("# class prob cx cy w h\n")

        for line in src:
            if not line.strip() or line.startswith("#"):
                continue

            left, right, *_ = line.split("|")
            name, prob = left.rsplit(None, 1)

            cls = names.index(name.strip())
            p = float(prob.strip().rstrip("%")) / 100.0
            box = [float(x) for x in right.split()[:4]]

            dst.write(
                f"{cls} {p:.8f} "
                + " ".join(f"{x:.8f}" for x in box)
                + "\n"
            )

if __name__ == "__main__":
    main()
