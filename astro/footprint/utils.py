from __future__ import absolute_import, division, print_function


class FootprintFile:

    def __init__(self, filename, format=None):
        """Initialize footprint file

        if ``format`` is not provided, it will be guessed from
        ``filename``
        """
        self.filename = filename
        self.format = format
        self.file = open(self.filename)


    def read(self):
        if self.format is None:
            ext = self.filename.split('.')[-1]
            if ext in ('dat', 'txt'):
                format = 'array'
            elif ext == 'reg':
                format = 'reg'
        self.file.close()


    def read_reg(self):
        for line in self.file:
            if line.startswith('polygon'):
                line = line[line.index('(')+1:line.index(')')].split(',')
                ra = np.array(line[::2], dtype=float)
                dec = np.array(line[1::2], dtype=float)
