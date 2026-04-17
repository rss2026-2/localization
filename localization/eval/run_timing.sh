bags=(
  "rosbags/simple_hall_all_topics_no_noise_1"
  "rosbags/simple_hall_all_topics_no_noise_2"
  "rosbags/simple_hall_all_topics_no_noise_3"
  "rosbags/simple_hall_all_topics_no_noise_4"
)

for bag in "${bags[@]}"
do
  echo "Processing $bag"

  # Extract just the folder name (run1, run2, etc.)
  bag_name=$(basename "$bag")

  # Start recording FIRST (important)
  ros2 bag record /timing/motion_model /timing/sensor_model -o "timing_${bag_name}_d" &
  RECORD_PID=$!

  sleep 1  # give recorder time to start

  # Play the bag
  ros2 bag play "$bag" &
  PLAY_PID=$!

  # Wait for playback to finish
  wait $PLAY_PID

  # Stop recording
  kill $RECORD_PID

  sleep 2  # allow clean shutdown

  echo "Finished $bag"
done
