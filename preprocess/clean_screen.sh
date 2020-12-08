set -e
grep_name=$1
echo "remove screen contains name $grep_name"
screen -ls | grep $grep_name
screen -ls | awk '{print $1}'| grep $grep_name | awk '{print "screen -S "$1" -X quit"}'| sh