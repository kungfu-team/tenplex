package vfs_test

import (
	"errors"
	"log"
	"strings"

	"github.com/kungfu-team/tenplex/mlfs/vfs"
)

func runScript(r *vfs.Tree, script string) error {
	cmds, err := parseCommands(script)
	if err != nil {
		return err
	}
	return runCommands(r, cmds)
}

func runCommands(r *vfs.Tree, cmds []cmd) error {
	for _, c := range cmds {
		log.Printf("%v", c)
		if err := c.run(r); err != nil {
			return err
		}
	}
	return nil
}

func parseCommands(text string) ([]cmd, error) {
	var cmds []cmd
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) == 2 {
			cmds = append(cmds, cmd{name: parts[0], arg: parts[1]})
		} else if len(parts) == 3 && parts[0] == `!` {
			cmds = append(cmds, cmd{name: parts[1], arg: parts[2], shouldFail: true})
		} else {
			return nil, errInvalidCommand
		}
	}
	return cmds, nil
}

type cmd struct {
	name, arg  string
	shouldFail bool
}

func (c *cmd) run(r *vfs.Tree) (err error) {
	switch c.name {
	case "mkdir":
		_, _, err = r.Mkdir(c.arg)
	case "touch":
		_, err = r.TouchText(c.arg, ``)
	case "rm":
		_, err = r.Rm(c.arg)
	case "rmdir":
		_, err = r.Rmdir(c.arg)
	default:
		err = errInvalidCommand
	}
	if c.shouldFail {
		if err != nil {
			return nil
		} else {
			return errShouldFail
		}
	}
	return err
}

var (
	errInvalidCommand = errors.New("invalid command")
	errShouldFail     = errors.New("should fail")
)
