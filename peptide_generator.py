from sequence import Sequence


def main():
    seq = Sequence('QKLSRAIAKGKDNLKEYKLNMS')
    pos_face, neg_face, positions = seq.get_faces(phi=180)
    print(seq)
    print(pos_face)
    print(neg_face)


if __name__ == '__main__':
    main()

