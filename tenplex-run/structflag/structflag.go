package structflag

import (
	"flag"
	"fmt"
	"log"
	"reflect"
	"strconv"
)

const (
	flagTag = `flag`
	defTag  = `default`
)

func RegisterFlags(v interface{}, flag *flag.FlagSet) {
	val := reflect.Indirect(reflect.ValueOf(v))
	for i := 0; i < val.NumField(); i++ {
		field := reflect.TypeOf(v).Elem().Field(i)
		if t := field.Tag.Get(flagTag); len(t) > 0 {
			vi := val.Field(i)
			registerFlag(&vi, t, field.Tag.Get(defTag), flag)
		}
	}
}

type FlagValue = flag.Value

var Log bool

func registerFlag(v *reflect.Value, name string, def string, flag *flag.FlagSet) {
	if Log {
		log.Printf("registerFlag %s with defval: %q", name, def)
	}
	switch k := v.Kind(); k {
	case reflect.String:
		p := (*string)(v.Addr().UnsafePointer())
		flag.StringVar(p, name, def, ``)
	case reflect.Int:
		p := (*int)(v.Addr().UnsafePointer())
		flag.IntVar(p, name, 0, ``)
		if len(def) > 0 {
			*p, _ = strconv.Atoi(def)
		}
	case reflect.Float64:
		p := (*float64)(v.Addr().UnsafePointer())
		flag.Float64Var(p, name, 0, ``)
		if len(def) > 0 {
			fmt.Sscanf(def, "%f", p)
		}
	case reflect.Bool:
		p := (*bool)(v.Addr().UnsafePointer())
		flag.BoolVar(p, name, false, ``)
		if len(def) > 0 {
			*p, _ = strconv.ParseBool(def)
		}
	default:
		if p, ok := v.Addr().Interface().(FlagValue); ok {
			flag.Var(p, name, ``)
			p.Set(def)
			return
		}
		log.Panicf("can't register flag for %s of %s", name, k)
	}
}

func ToGoArgs(v interface{}) []string {
	return ToArgs(v, ShortFlagName)
}

func ToArgs(v interface{}, toFlag flagName) []string {
	var args []string
	val := reflect.Indirect(reflect.ValueOf(v))
	for i := 0; i < val.NumField(); i++ {
		field := reflect.TypeOf(v).Elem().Field(i)
		if t := field.Tag.Get(flagTag); len(t) > 0 {
			vi := val.Field(i)
			args = append(args, toArg(&vi, t, toFlag)...)
		}
	}
	return args
}

func toArg(v *reflect.Value, name string, toFlag flagName) []string {
	flg := toFlag(name)
	switch v.Kind() {
	case reflect.String:
		p := (*string)(v.Addr().UnsafePointer())
		return []string{flg, *p}
	case reflect.Int:
		p := (*int)(v.Addr().UnsafePointer())
		return []string{flg, str(*p)}
	case reflect.Float64:
		p := (*float64)(v.Addr().UnsafePointer())
		return []string{flg, fmt.Sprintf("%f", *p)}
	case reflect.Bool:
		p := (*bool)(v.Addr().UnsafePointer())
		if *p {
			return []string{flg}
		} else {
			return nil
		}
	default:
		return nil
	}
}

func ShortFlagName(name string) string { return `-` + name }

func LongFlagName(name string) string { return `--` + name }

var str = strconv.Itoa

type flagName func(string) string
