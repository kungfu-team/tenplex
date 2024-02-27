package structflag

import (
	"flag"
	"log"
	"reflect"
	"strconv"
)

const tagName = `flag`

func RegisterFlags(v interface{}, flag *flag.FlagSet) {
	val := reflect.Indirect(reflect.ValueOf(v))
	for i := 0; i < val.NumField(); i++ {
		field := reflect.TypeOf(v).Elem().Field(i)
		if t := field.Tag.Get(tagName); len(t) > 0 {
			vi := val.Field(i)
			registerFlag(&vi, t, flag)
		}
	}
}

type FlagValue = flag.Value

func registerFlag(v *reflect.Value, name string, flag *flag.FlagSet) {
	switch v.Kind() {
	case reflect.String:
		p := (*string)(v.Addr().UnsafePointer())
		flag.StringVar(p, name, ``, ``)
	case reflect.Int:
		p := (*int)(v.Addr().UnsafePointer())
		flag.IntVar(p, name, 0, ``)
	default:
		if p, ok := v.Interface().(FlagValue); ok {
			flag.Var(p, name, ``)
			return
		}
		log.Printf("can't register flag")
	}
}

func ToFlags(v interface{}) []string {
	var args []string
	val := reflect.Indirect(reflect.ValueOf(v))
	for i := 0; i < val.NumField(); i++ {
		field := reflect.TypeOf(v).Elem().Field(i)
		if t := field.Tag.Get(tagName); len(t) > 0 {
			vi := val.Field(i)
			args = append(args, toArg(&vi, t)...)
		}
	}
	return args
}

func toArg(v *reflect.Value, name string) []string {
	switch v.Kind() {
	case reflect.String:
		p := (*string)(v.Addr().UnsafePointer())
		return []string{`-` + name, *p}
	case reflect.Int:
		p := (*int)(v.Addr().UnsafePointer())
		return []string{`-` + name, str(*p)}
	default:
		return nil
	}
}

var str = strconv.Itoa
