-- Table of valid chunks with dolphin calls
SELECT *
FROM (
    SELECT
        am.*,
        l.label AS label,
        l.call_begin_time AS call_begin_time,
        l.call_end_time AS call_end_time,
        l.call_length_seconds AS call_length_seconds,
        l.call_channel AS call_channel,
        CASE
            WHEN (
                l.call_begin_time < am.chunk_end_seconds AND
                am.chunk_end_seconds < l.call_end_time
            ) THEN (am.chunk_end_seconds - GREATEST(l.call_begin_time, am.chunk_start_seconds))
            WHEN (
                l.call_begin_time < am.chunk_start_seconds AND
                am.chunk_start_seconds < l.call_end_time
            ) THEN (LEAST(l.call_end_time, am.chunk_end_seconds) - am.chunk_start_seconds)
            ELSE
                l.call_length_seconds END AS call_length_within_chunk
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
    QUALIFY
        row_number() OVER (PARTITION BY am.audio_filename, am.chunk_index ORDER BY l.call_begin_time) = 1
) AS t
WHERE
    t.call_length_seconds = t.call_length_within_chunk OR -- 100% of call length within chunk to consider small calls
    t.call_length_within_chunk >= {minimum_percentage_of_call_in_chunk} * (t.chunk_end_seconds - t.chunk_start_seconds)

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
