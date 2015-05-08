$(document).ready(function () {
	$("button").click(onContinue);
	$("#quizForm, .quizEnd, #tryAgain").hide();
    $(".question").addClass("questionHider");
});

function onContinue () {
    var value = $(this).val();
    switch (value) {
        case "onStart":
            $("#startButton").hide();
            $("#quizForm").show();
            $("#q1").removeClass("questionHider");
            break;
        case "onSubmit":
            var last = "4";
            if (correctAnswer(last)) {
                $("#tryAgain").hide();
                $(this).parent().addClass("questionHider");
                $(".question").hide()
                $(".quizEnd").show();
            } else {
                var tryAgainText = "You call yourself a mother? Try again.", wrongBefore = false;
                $("#tryAgain").text(tryAgainText);
                if ($("#tryAgain").attr("display") !== "none") {
                    wrongBefore = true;
                }
                if (wrongBefore) {
                    $("#tryAgain").fadeOut("fast");
                    $("#tryAgain").fadeIn("fast");
                } else {
                    $("#tryAgain").show();
                }
            }
            break;
        default:
            if (correctAnswer(value)) {
                $("#tryAgain").hide();
                $("#q" + value).addClass("questionHider");
                $("#q" + (1 + parseInt(value))).removeClass("questionHider");
            } else {
                var tryAgainText, wrongBefore = false;
                if (value === "1") {
                    tryAgainText = "Then this webpage is not for you. Try again.";
                } else {
                    tryAgainText = "You call yourself a mother? Try again.";
                }
                $("#tryAgain").text(tryAgainText);
                if ($("#tryAgain").attr("display") !== "none") {
                    wrongBefore = true;
                }
                if (wrongBefore) {
                    $("#tryAgain").fadeOut("fast");
                    $("#tryAgain").fadeIn("fast");
                } else {
                    $("#tryAgain").show();
                }
            }
            break;
    }
}

function correctAnswer (value) {
    switch (value) {
        case "1":
            return $("#q1 input:radio:checked").val() === "yes1";
        case "2":
            return $("#q2 input:text").val().toLowerCase() === "manohar jois";
        case "3":
            var $checkboxGroup = $("#q3 input:checkbox").filter(":checked")
            return $checkboxGroup.length === 5;
        case "4":
            return $("#q4 input:radio:checked").val() === "yes4";
        default:
            return true;
    }
}
