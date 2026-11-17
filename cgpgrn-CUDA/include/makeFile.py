import os
import argparse

def get_parser() -> argparse.ArgumentParser:
    '''
    :return: an argparse ArgumentParser object for parsing command
        line parameters
    '''
    parser = argparse.ArgumentParser(
        description='CUDA Makefile generator for CGPGRN.',
        epilog='Example usage: python makeFile_cuda.py -o 10 -n 100 -e 50000'
    )

    parser.add_argument('-o', dest='MAXOUT', type=str,
                        default='None',
                        help='Maximum number of outputs \n'
                        )

    parser.add_argument('-n', dest='MAXNODE', type=str,
                        default='None',
                        help='Maximum number of nodes \n'
                        )

    parser.add_argument('-e', dest='MAXEVAL', type=str,
                        default='None',
                        help='Maximum number of evaluations \n'
                        )

    return parser


def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts


def make(MAX_OUTPUTS, MAX_NODES, MAX_EVAL):

    makeFileName = 'Makefile'
    if os.name == 'nt':
        cudaPath = os.environ.get('CUDA_PATH', '').replace('\\', '/')
        cudaLib = cudaPath + '/lib/x64'
    else:
        cudaPath = os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', '/usr/local/cuda'))
        cudaLib = cudaPath + '/lib64'

    with open(makeFileName, 'w') as makeFileOpen:
        makeFileOpen.write("CC = g++\n")
        makeFileOpen.write("CFLAGS = -O3 -Wall\n\n")
        makeFileOpen.write("MAXOUTVALUE = " + str(MAX_OUTPUTS) + "\n")
        makeFileOpen.write("MAXNODESVALUE = " + str(MAX_NODES) + "\n")
        makeFileOpen.write("MAXEVAL = " + str(MAX_EVAL) + "\n\n")
        makeFileOpen.write("main:\n")

        defines = "-DMAX_OUTPUTS=${MAXOUTVALUE} -DMAX_NODES=${MAXNODESVALUE} -DNUM_GENERATIONS=${MAXEVAL}"
        if os.name == 'nt':
            cuda = f'-I"{cudaPath}/include" -L"{cudaLib}" -lnvrtc -lcuda'
        else:
            # lib64/stubs: garante o libcuda.so para o link mesmo se o driver so instalar libcuda.so.1
            # rpath: o executavel acha o libnvrtc.so sem precisar de LD_LIBRARY_PATH
            cuda = f'-I"{cudaPath}/include" -L"{cudaLib}" -L"{cudaLib}/stubs" -Wl,-rpath,"{cudaLib}" -lnvrtc -lcuda'
        output = 'progCUDA.exe' if os.name == 'nt' else 'progCUDA'

        makeFileOpen.write(f'\t${{CC}} ${{CFLAGS}} {defines} source/*.cpp -o {output} {cuda}\n')

if __name__ == '__main__':
    opts = parse_arguments()
    MAXOUT = opts.MAXOUT
    MAXNODE = opts.MAXNODE
    MAXEVAL = opts.MAXEVAL
    make(MAXOUT, MAXNODE, MAXEVAL)

    if os.name != 'nt':
        os.system('chmod +x Makefile')
        os.system('make')
    else:
        os.system('cmd /c "make"')
