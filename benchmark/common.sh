
join_() {
    local IFS=$1
    shift
    echo "$*"
}

join() { join_ , $@; }

