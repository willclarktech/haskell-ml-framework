testfiles="./*.test.hs"

for f in $testfiles
	do
		echo "Running $f"
		runghc -- -fno-warn-tabs "$f"
	done

