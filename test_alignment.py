from alignment.temporal_alignment import align_features


def test_temporal_alignment():

    streams = {

        "audio": [
            {
                "timestamp":"10:00",
                "energy":0.5
            }
        ],

        "imu": [
            {
                "timestamp":"10:00",
                "movement":0.2
            }
        ]
    }


    result = align_features(streams)


    assert len(result) == 1
    assert result[0]["audio_energy"] == 0.5
    assert result[0]["imu_movement"] == 0.2