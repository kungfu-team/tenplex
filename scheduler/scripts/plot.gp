#!/usr/bin/env gnuplot

set terminal postscript portrait dashed color size 14,9 font 44 fontscale 1
# set terminal postscript portrait dashed monochrome size 14,9 font 44 fontscale 1
set datafile missing '-'
set boxwidth 0.9 absolute

set style fill solid 1.00 border lt -1
set style data lines

set xtics ()
set xtics border in scale 0,0 nomirror autojustify
set xtics nomirror rotate by -45
set xtics norangelimit

set key fixed right bottom vertical Right noreverse noenhanced autotitle nobox
set key outside;

NO_ANIMATION = 1

set output 'p1.ps'

f1 = 'a.log'
f2 = 'b.log'

plot f1 using ($1 / 60.0):2 title 'a loss', \
     f2 using ($1 / 60.0):2 title 'b loss' \
