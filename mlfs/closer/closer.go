package closer

import "io"

type closer struct {
	r     io.Reader
	close func() error
}

func ReadClose(r io.Reader, close func() error) io.ReadCloser {
	return &closer{r: r, close: close}
}

func (c *closer) Read(buf []byte) (int, error) {
	return c.r.Read(buf)
}

func (c *closer) Close() error {
	return c.close()
}
