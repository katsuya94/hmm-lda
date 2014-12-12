for file in `ls -1 | grep '^document\d\+.dat$'`
do
    e=filename=\'${file}\'
    gnuplot -e $e plot.plt
done
gnuplot -e "filename='classtotals.dat'" plot.plt
gnuplot -e "filename='topictotals.dat'" plot.plt
