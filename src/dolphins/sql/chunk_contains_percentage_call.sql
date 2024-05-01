-- Table of valid chunks with dolphin calls
SELECT
    am.*,
    FIRST(l.labels_df ORDER BY l.call_begin_time) AS label,
    FIRST(l.call_begin_time ORDER BY l.call_begin_time) AS call_begin_time,
    FIRST(l.call_end_time ORDER BY l.call_begin_time) AS call_end_time,
    FIRST(l.call_length_seconds ORDER BY l.call_begin_time) AS call_length_seconds,
    FIRST(l.call_channel ORDER BY l.call_begin_time) AS call_channel,
    FIRST(
        CASE WHEN (l.call_begin_time < am.chunk_end_seconds AND
            am.chunk_end_seconds < l.call_end_time) THEN (am.chunk_end_seconds - GREATEST(l.call_begin_time, am.chunk_start_seconds))
        CASE WHEN (l.call_begin_time < am.chunk_start_seconds AND
            am.chunk_start_seconds < l.call_end_time) THEN (LEAST(l.call_end_time, am.chunk_end_seconds) -am.chunk_start_seconds)
        ELSE l.call_length_seconds
        ORDER BY l.call_begin_time) AS call_lenght_within_chunk,
    COUNT(*) AS label_count_within_chunk
FROM
    audio_metadata_df AS am INNER JOIN
    labels_df AS l 
        ON (( --  Keep chunks that contain a call within its boundaries
            am.audio_filename = l.audio_filename AND
            am.chunk_start_seconds <= l.call_begin_time AND
            l.call_end_time <= am.chunk_end_seconds
        ) OR ( -- Keep chunks that contain only the ending of a call
            am.audio_filename = l.audio_filename AND
            l.call_begin_time < am.chunk_start_seconds AND
            am.chunk_start_seconds < l.call_end_time
        ) OR ( -- Keep chunks that contain only the beginning of a call
            am.audio_filename = l.audio_filename AND
            l.call_begin_time < am.chunk_end_seconds AND
            am.chunk_end_seconds < l.call_end_time
        ))
WHERE
    l.call_length_seconds = call_lenght_within_chunk OR
    call_lenght_within_chunk >= 0.6 * (am.chunk_end_seconds - am.chunk_start_seconds) -- 60% call lenght
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
