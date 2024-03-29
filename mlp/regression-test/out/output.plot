# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6
set style line 2 lt rgb "#00FF00" lw 3 pt 6
set style line 3 lt rgb "#0000FF" lw 3 pt 6

# plot data
plot "output.dat" using 1:3 with linespoints ls 1 title 'MLP',\
     "output.dat" using 1:1 with linespoints ls 2 title 'IN',\
     "output.dat" using 1:2 with linespoints ls 3 title 'TRAIN'
