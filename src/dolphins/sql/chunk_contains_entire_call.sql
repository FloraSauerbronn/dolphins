-- Table of valid chunks with dolphin calls
SELECT
    am.*,
    FIRST(l.label ORDER BY l.call_begin_time) AS label,
    FIRST(l.call_channel ORDER BY l.call_begin_time) AS call_channel,
    FIRST(l.call_begin_time ORDER BY l.call_begin_time) AS call_begin_time,
    FIRST(l.call_end_time ORDER BY l.call_begin_time) AS call_end_time,
    FIRST(l.call_length_seconds ORDER BY l.call_begin_time) AS call_length_seconds,
    COUNT(*) AS label_count_whithin_chunk
FROM
    audio_metadata_df AS am LEFT JOIN
    labels_df AS l 
        ON am.audio_filename = l.audio_filename AND
        am.chunk_start_seconds <= l.call_begin_time AND
        l.call_end_time <= am.chunk_end_seconds
WHERE
    l.label IS NOT NULL
GROUP BY am.*

UNION BY NAME

-- Table of valid chunks with no dolphin calls
SELECT
    *,
    'no_call' AS label,
FROM
    audio_metadata_df AS am ANTI JOIN
    labels_df AS l 
        ON (( -- Remove chunks that contain a call within its boundaries
            am.audio_filename = l.audio_filename AND
            am.chunk_start_seconds <= l.call_begin_time AND
            l.call_end_time <= am.chunk_end_seconds
        ) OR ( -- Remove chunks that contain only the ending of a call
            am.audio_filename = l.audio_filename AND
            l.call_begin_time < am.chunk_start_seconds AND
            am.chunk_start_seconds < l.call_end_time
        ) OR ( -- Remove chunks that contain only the beginning of a call
            am.audio_filename = l.audio_filename AND
            l.call_begin_time < am.chunk_end_seconds AND
            am.chunk_end_seconds < l.call_end_time
        ))
