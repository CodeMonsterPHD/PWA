def generate_identity_3DLUT(dim, output_file):
    step = 1.0 / (dim - 1)
    with open(output_file, 'w') as f:
        for k in range(dim):
            for j in range(dim):
                for i in range(dim):
                    f.write('{:.6f}  {:.6f}  {:.6f}\n'.format(
                        step * i, step * j, step * k))


if __name__ == '__main__':
    output_file = 'IdentityLUT{}.txt'.format(49)
    generate_identity_3DLUT(65, output_file)