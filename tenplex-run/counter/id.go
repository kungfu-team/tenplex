package counter

func NewCounterFunc() func() int {
	var id int
	return func() int { x := id; id++; return x }
}

func New() *Counter {
	return &Counter{}
}

type Counter struct {
	n int
}

func (c *Counter) Next() int {
	id := c.n
	c.n++
	return id
}

func (c *Counter) Reset() {
	c.n = 0
}
