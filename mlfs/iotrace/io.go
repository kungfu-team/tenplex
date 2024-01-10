package iotrace

import "io"

type TracedWriter struct {
	w io.Writer
	c *Counter
}

func TraceWriter(w io.Writer, c *Counter) io.Writer {
	return &TracedWriter{
		w: w,
		c: c,
	}
}

func (w *TracedWriter) Write(bs []byte) (int, error) {
	n, err := w.w.Write(bs)
	w.c.Add(int64(n))
	return n, err
}

type TracedReader struct {
	r io.Reader
	c *Counter
}

func TraceReader(r io.Reader, c *Counter) io.Reader {
	if c == nil {
		return r
	}
	return &TracedReader{
		r: r,
		c: c,
	}
}

func (r *TracedReader) Read(bs []byte) (int, error) {
	n, err := r.r.Read(bs)
	r.c.Add(int64(n))
	return n, err
}
