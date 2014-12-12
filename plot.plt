set term png
set output filename.".png"
plot filename using 1 with lines title "0", \
     filename using 2 with lines title "1", \
     filename using 3 with lines title "2", \
     filename using 4 with lines title "3"
