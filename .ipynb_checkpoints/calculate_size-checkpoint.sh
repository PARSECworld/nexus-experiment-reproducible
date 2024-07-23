#!/bin/bash

# Function to get size of a file
get_file_size() {
    stat --printf="%s" "$1"
}

# Initialize total size counter
total_size=0

# Create an array to store file sizes and paths
declare -a file_sizes

# List all files in the repository, handling spaces and special characters
while IFS= read -r -d '' file; do
    # Check if the file is ignored by git
    if ! git check-ignore -q "$file"; then
        # If not ignored, add its size to the total
        file_size=$(get_file_size "$file")
        total_size=$((total_size + file_size))
        # Store file size and path in the array
        file_sizes+=("$file_size:$file")
    fi
done < <(find . -type f -print0)

# Convert total size to human-readable format
total_size_hr=$(echo "$total_size" | awk '{ split( "B KB MB GB TB", v ); s=1; while( $1>1024 ){ $1/=1024; s++ } printf "%.2f %s", $1, v[s] }')

# Write file sizes to file_sizes.txt
{
    for entry in "${file_sizes[@]}"; do
        size=${entry%%:*}
        path=${entry#*:}
        size_hr=$(echo "$size" | awk '{ split( "B KB MB GB TB", v ); s=1; while( $1>1024 ){ $1/=1024; s++ } printf "%.2f %s", $1, v[s] }')
        echo "$size_hr - $path"
    done | sort -rh

    echo "Total size of tracked files: $total_size_hr"
} > file_sizes.txt

echo "File sizes have been written to file_sizes.txt"