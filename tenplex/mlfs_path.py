class MLFSPath(object):

    def __init__(self, mnt='/mnt/mlfs') -> None:
        self.mnt = mnt

    def _path(self, p):
        return self.mnt + p

    def _read_lines(self, filename):
        return [line.strip() for line in open(self._path(filename))]

    def filenames(self, job, rank):
        lines = self._read_lines
        head = lines(f'/job/{job}/head.txt')[0]
        part = lines(head)[rank]
        names = lines(f'{part}/list.txt')
        return [self._path(n) for n in names]
