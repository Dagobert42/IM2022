

lock_daytime = """
local timer = 0
minetest.register_globalstep(function(dtime)
    timer = timer + dtime;
    if timer >= 30 then
        minetest.set_timeofday(0.7)
        timer = 0
    end
end)
"""