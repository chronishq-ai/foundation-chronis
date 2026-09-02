from typing import List, Dict, Any


def align_features(
    feature_streams: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Align multiple feature streams using timestamps.

    Input example:

    {
        "audio":[
            {"timestamp":"10:00","energy":0.5}
        ],

        "imu":[
            {"timestamp":"10:00","movement":0.2}
        ]
    }

    Output:

    [
        {
          "timestamp":"10:00",
          "audio_energy":0.5,
          "movement":0.2
        }
    ]
    """

    aligned = {}

    for stream_name, records in feature_streams.items():

        for record in records:

            timestamp = record["timestamp"]

            if timestamp not in aligned:
                aligned[timestamp] = {
                    "timestamp": timestamp
                }

            for key, value in record.items():

                if key != "timestamp":

                    aligned[timestamp][
                        f"{stream_name}_{key}"
                    ] = value


    return list(aligned.values())