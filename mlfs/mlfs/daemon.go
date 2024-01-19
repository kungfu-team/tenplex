package mlfs

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/kungfu-team/tenplex/mlfs/buildinfo"
	"github.com/kungfu-team/tenplex/mlfs/pid"
	"github.com/kungfu-team/tenplex/mlfs/utils"
)

const DefaultCtrlPort = 20010

type Daemon struct {
	CtrlPort   int
	HTTPPort   int
	Mount      string
	Tmp        string
	Super      bool
	Token      string
	Host       string
	Redundancy int
	Peers      string
	LogReq     bool
	Info       bool
}

func (d *Daemon) RegisterFlags(flag *flag.FlagSet) {
	flag.IntVar(&d.CtrlPort, `ctrl-port`, 0, ``)
	flag.IntVar(&d.HTTPPort, `http-port`, 0, ``)
	flag.StringVar(&d.Mount, `mnt`, ``, ``)
	flag.StringVar(&d.Tmp, `tmp`, ``, ``)
	flag.BoolVar(&d.Super, `su`, false, ``)
	flag.BoolVar(&d.LogReq, `log-req`, false, ``)
	flag.BoolVar(&d.Info, `info`, false, ``)
	flag.StringVar(&d.Token, `token`, ``, ``)
	flag.StringVar(&d.Host, `host`, ``, `host to listen`)
	flag.IntVar(&d.Redundancy, `r`, 0, `number of extra copies to write`)
	flag.StringVar(&d.Peers, `peers`, ``, `comma separated list of peer ids`)
}

func (d Daemon) Run() error {
	d.RunCtx(context.TODO())
	return nil
}

var (
	errInvalidRedundancy = errors.New(`redundency < len(peers) is required`)
)

func (d Daemon) RunCtx(ctx context.Context) {
	if d.Info {
		buildinfo.Default.Show(os.Stdout)
	}
	e := New()
	e.LogReq = d.LogReq
	if len(d.Peers) > 0 {
		ids, err := pid.ParsePeerList(d.Peers)
		if err != nil {
			utils.ExitErr(err)
		}
		e.peers = ids
	}
	if d.Redundancy > 0 {
		if e.redundency >= len(d.Peers) {
			utils.ExitErr(errInvalidRedundancy)
		}
		e.redundency = d.Redundancy
		e.rank, _ = e.peers.Rank(pid.PeerID{
			IPv4: pid.MustParseIPv4(d.Host),
			Port: uint16(d.CtrlPort),
		})
	}
	tree := e.Tree()
	{
		tree.TouchText(`/git-commit`, fmt.Sprintln(buildinfo.Default.GitCommit))
		tree.TouchText(`/build-timestamp`, fmt.Sprintln(buildinfo.Default.BuildTimestamp))
	}
	if len(d.Token) > 0 {
		tree.TouchText(`/token.txt`, fmt.Sprintf("%s\n", d.Token))
	}
	if d.CtrlPort > 0 {
		tree.TouchText(`/ctrl-port.txt`, fmt.Sprintf("%d\n", d.CtrlPort))
	}
	if d.HTTPPort > 0 {
		tree.TouchText(`/http-port.txt`, fmt.Sprintf("%d\n", d.HTTPPort))
	}
	if len(d.Tmp) > 0 {
		if err := os.MkdirAll(d.Tmp, os.ModePerm); err != nil {
			utils.ExitErr(err)
		}
		if err := e.SetCache(d.Tmp); err != nil {
			utils.ExitErr(err)
		}
		log.Printf("using cache dir: %s", d.Tmp)
	}
	var wg sync.WaitGroup
	if d.CtrlPort > 0 {
		wg.Add(1)
		go func(port int) {
			defer wg.Done()
			e.RunCtrl(port)
		}(d.CtrlPort)
	}
	if d.HTTPPort > 0 {
		wg.Add(1)
		go func(port int) {
			defer wg.Done()
			e.RunHTTP(port)
		}(d.HTTPPort)
		if !WaitTCP(d.Host, d.HTTPPort) {
			panic("start HTTP server failed")
		}
	}
	if len(d.Mount) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e.RunFuse(d.Mount, d.Super)
		}()
	}
	trap(func(os.Signal) {
		e.Stop()
	})
	go func() {
		<-ctx.Done()
		e.Stop()
	}()
	wg.Wait()
}

func trap(cancel func(os.Signal)) {
	c := make(chan os.Signal)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-c
		cancel(sig)
	}()
}

func WaitTCP(host string, port int) bool {
	t0 := time.Now()
	for {
		if time.Since(t0) > 5*time.Second {
			return false
		}
		if _, err := net.Dial("tcp", net.JoinHostPort(host, strconv.Itoa(port))); err == nil {
			break
		}
		time.Sleep(1 * time.Second)
	}
	log.Printf("tcp://%s:%d is up", host, port)
	return true
}
