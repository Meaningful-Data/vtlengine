from vtlengine import prettify


class TestViralPropagationPrettify:
    def test_prettify_enumerated(self):
        script = (
            "define viral propagation CONF_priority(variable At_1) is "
            'when "C" then "C";when "N" then "N";else "F" '
            "end viral propagation;"
            "DS_r<-DS_1+DS_2;"
        )
        result = prettify(script=script)
        assert "define viral propagation" in result
        assert 'when "C" then "C"' in result
        assert 'else "F"' in result
        assert "end viral propagation" in result

    def test_prettify_aggregate(self):
        script = (
            "define viral propagation TIME_prop(variable At_1)"
            "is aggr max end viral propagation;"
            "DS_r<-DS_1;"
        )
        result = prettify(script=script)
        assert "define viral propagation" in result
        assert "aggr max" in result
        assert "end viral propagation" in result

    def test_prettify_binary_clause(self):
        script = (
            "define viral propagation MIX(variable At_1) is "
            'when "C" and "M" then "N";when "M" then "M";else " " '
            "end viral propagation;"
            "DS_r<-DS_1;"
        )
        result = prettify(script=script)
        assert 'when "C" and "M" then "N"' in result

    def test_prettify_valuedomain(self):
        script = (
            "define viral propagation OBS(valuedomain CL_OBS)"
            'is when "M" then "M";else "A" '
            "end viral propagation;"
            "DS_r<-DS_1;"
        )
        result = prettify(script=script)
        assert "valuedomain CL_OBS" in result
