package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"sort"

	"github.com/kungfu-team/mlfs/buildinfo"
	"github.com/kungfu-team/mlfs/mlfs"
)

var config = flag.String("config", ``, ``)

func main() {
	flag.Parse()
	if len(os.Args) == 0 {
		return
	}
	prog := os.Args[0]
	args := flag.Args()
	if len(args) == 0 {
		usage(path.Base(prog))
		return
	}
	cmd, args := args[0], args[1:]
	if f, ok := commands[cmd]; ok {
		f(args)
		return
	}
	usage(path.Base(prog))
}

var commands = map[string]func([]string){
	`mount`:  func(args []string) { runSubCmd(`mount`, &MountCmd{}, args) },
	`fetch`:  func(args []string) { runSubCmd(`fetch`, &FetchCmd{}, args) },
	`relay`:  func(args []string) { runSubCmd(`relay`, &mlfs.RelayServer{}, args) },
	`bench`:  func(args []string) { runSubCmd(`bench`, &mlfs.Test{}, args) },
	`serve`:  func(args []string) { runSubCmd(`serve`, &mlfs.LocalServer{}, args) },
	`daemon`: func(args []string) { runSubCmd(`daemon`, &mlfs.Daemon{}, args) },
	`info`:   info,
}

func usage(prog string) {
	fmt.Println("Usage:")
	var cmds []string
	for c := range commands {
		cmds = append(cmds, c)
	}
	sort.Strings(cmds)
	for _, c := range cmds {
		fmt.Printf("\t%s %s\n", prog, c)
	}
}

type MountCmd struct {
	ClientFlags
	IdxName         string
	IdxFile         string
	JobID           string
	Progress        int
	DpSize          int
	GlobalBatchSize int
	Seed            int
}

func (c *MountCmd) RegisterFlags(flag *flag.FlagSet) {
	c.ClientFlags.RegisterFlags(flag)
	flag.StringVar(&c.IdxName, `idx-name`, ``, ``)
	flag.StringVar(&c.IdxFile, `index-url`, ``, ``)
	flag.StringVar(&c.JobID, `job`, `0`, ``)
	flag.IntVar(&c.Progress, `progress`, 0, ``)
	flag.IntVar(&c.GlobalBatchSize, `global-batch-size`, 1, ``)
	flag.IntVar(&c.DpSize, `dp-size`, 1, ``)
	flag.IntVar(&c.Seed, `seed`, 0, ``)
}

func (m MountCmd) Run() error {
	if !mlfs.WaitTCP(m.Host, m.CtrlPort) {
		return errWaitTimeout
	}
	cli, err := mlfs.NewClientTo(m.Host, m.CtrlPort)
	if err != nil {
		return err
	}
	if err := cli.AddIndex(m.IdxName, m.IdxFile); err != nil {
		return err
	}
	if err := cli.Mount(m.JobID, m.IdxName, int64(m.Progress), m.GlobalBatchSize, m.DpSize, m.Seed); err != nil {
		return err
	}
	var s string
	if err := cli.GetRoot(&s); err != nil {
		return err
	}
	fmt.Printf("mounted at: %s\n", s)
	return nil
}

type FetchCmd struct {
	ClientFlags
	File string
	MD5  string
}

func (c *FetchCmd) RegisterFlags(flag *flag.FlagSet) {
	c.ClientFlags.RegisterFlags(flag)
	flag.StringVar(&c.File, `file`, ``, ``)
	flag.StringVar(&c.MD5, `md5`, ``, ``)
}

func (c FetchCmd) Run() error {
	cli, err := mlfs.NewClientTo(c.Host, c.CtrlPort)
	if err != nil {
		return err
	}
	return cli.Fetch(c.File, c.MD5)
}

type ClientFlags struct {
	Host     string
	CtrlPort int
}

func (c *ClientFlags) RegisterFlags(flag *flag.FlagSet) {
	flag.StringVar(&c.Host, `host`, ``, ``)
	flag.IntVar(&c.CtrlPort, `ctrl-port`, mlfs.DefaultCtrlPort, ``)
}

func info(args []string) {
	var c ClientFlags
	flag := flag.NewFlagSet(`info`, flag.ExitOnError)
	c.RegisterFlags(flag)
	flag.Parse(args)
	panicErr(showInfo(os.Stdout, &c))
}

func showInfo(w io.Writer, c *ClientFlags) error {
	fmt.Fprintln(w, "Client info:")
	buildinfo.Default.Show(w)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Server info:")
	if !mlfs.WaitTCP(c.Host, c.CtrlPort) {
		return errWaitTimeout
	}
	resp, err := http.Get(fmt.Sprintf("http://%s:%d/debug", c.Host, c.CtrlPort))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	io.Copy(w, resp.Body)
	fmt.Fprintln(w)
	return nil
}

func panicErr(err error) {
	if err != nil {
		panic(err)
	}
}

var errWaitTimeout = errors.New(`wait timeout`)

type SubCmd interface {
	RegisterFlags(flag *flag.FlagSet)
	Run() error
}

func runSubCmd(name string, c SubCmd, args []string) {
	flag := flag.NewFlagSet(name, flag.ExitOnError)
	c.RegisterFlags(flag)
	flag.Parse(args)
	panicErr(c.Run())
}
