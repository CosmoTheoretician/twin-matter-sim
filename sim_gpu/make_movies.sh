#!/bin/bash
DESKTOP=/mnt/c/Users/roche/Desktop/movies
FPS=4
mkdir -p "$DESKTOP"

find out -name interpretation.md -printf '%h\n' | while read -r d; do
    name=$(basename "$d")
    list="$d/frames.txt"
    find "$d" -name '*snap*.png' | sort | while read -r f; do
        echo "file '$(realpath "$f")'"
        echo "duration $(awk "BEGIN{printf \"%.3f\", 1/$FPS}")"
    done > "$list"

    count=$(grep -c '^file' "$list")
    if [ "$count" -eq 0 ]; then
        echo "No snapshots in $d, skipping"
        continue
    fi
    tail -n 2 "$list" | head -n 1 >> "$list"  # repeat last frame so it is shown

    echo "Making movie for $name ($count frames)"
    ffmpeg -y -loglevel error -f concat -safe 0 -i "$list" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -r 25 \
        "$d/movie.mp4" < /dev/null
    cp "$d/movie.mp4" "$DESKTOP/$name.mp4"
done
echo "Done. Movies are in $DESKTOP"