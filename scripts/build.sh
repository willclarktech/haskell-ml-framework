outputdir="./build"
bindir="./bin"
inputfile="$1"
outputfile=${inputfile%.hs}

ghc -fno-warn-tabs -O2 -outputdir="$outputdir" "$inputfile" -o "$bindir/$outputfile"
